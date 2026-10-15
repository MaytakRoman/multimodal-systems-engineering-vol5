import os,sys,json,time,hashlib,platform,resource,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HUB_DISABLE_XET']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import CLIPModel,CLIPProcessor,WhisperProcessor,WhisperForConditionalGeneration
import soundfile as sf
from scipy.signal import resample_poly
import urllib.request
MODELS=json.loads((ROOT/'models.json').read_text());OUT=ROOT/'review_results/open_models';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1);torch.manual_seed(2026)
def save(name,o):(OUT/name).write_text(json.dumps(o,indent=2,ensure_ascii=False))
meta=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,transformers=__import__('transformers').__version__,threads=1,platform=platform.platform(),processor=platform.processor(),cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].strip(': '))
name='openai/clip-vit-base-patch32';rev=MODELS[name]['sha']
print('download CLIP',flush=True)
folder=snapshot_download(name,revision=rev,cache_dir=str(ROOT/'model_cache'),allow_patterns=['*.json','*.txt',('model.safetensors' if 'model.safetensors' in MODELS[name]['files'] else 'pytorch_model.bin')])
print('load CLIP',flush=True)
model=CLIPModel.from_pretrained(folder,local_files_only=True).eval();processor=CLIPProcessor.from_pretrained(folder,local_files_only=True)
# Three actual photographs shipped in a versioned scientific image package.
labels=['astronaut','coffee','cat'];files=['astronaut.png','coffee.png','chelsea.png'];images=[]
for name in files:
 import skimage
 src=Path(skimage.__file__).parent/'data'/name;dst=OUT/name;dst.write_bytes(src.read_bytes());images.append(Image.open(dst).convert('RGB'))
texts=['a photo of an astronaut','a photo of a cup of coffee','a photo of a cat']
inputs=processor(text=texts,images=images,return_tensors='pt',padding=True)
with torch.inference_mode():
 out=model(**inputs);scores=out.logits_per_image.cpu().numpy();vi=out.image_embeds.cpu().numpy();vt=out.text_embeds.cpu().numpy()
cos=vi@vt.T;pred=np.argmax(cos,axis=1);retrieval=dict(model='openai/clip-vit-base-patch32',revision=rev,labels=labels,texts=texts,cosine=cos.tolist(),logits=scores.tolist(),top1=pred.tolist(),correct=int(sum(pred==np.arange(3))),n=3,source='scikit-image 0.25.2 bundled astronaut.png, coffee.png, chelsea.png',image_sha256={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in files},meta=meta)
save('clip_retrieval.json',retrieval);print('CLIP retrieval',pred,flush=True)
# A second task: image-only inference latency on this CPU, not an accelerator claim.
pix=processor(images=images[0],return_tensors='pt')['pixel_values']
with torch.inference_mode():
 for _ in range(5):model.get_image_features(pixel_values=pix)
 times=[]
 for _ in range(40):
  start=time.perf_counter_ns();model.get_image_features(pixel_values=pix);times.append((time.perf_counter_ns()-start)/1e6)
profile=dict(operation='CLIP image encoder, one 224x224 RGB image, batch=1, FP32, CPU',warmup=5,repeats=40,latency_ms=times,p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),max_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),rss_scope='process high-water mark since start, includes loading both CLIP encoders; not incremental tensor memory',meta=meta,model_revision=rev)
save('cpu_profile.json',profile);print('profile',profile['p50_ms'],profile['p95_ms'],flush=True)
del model,processor,inputs,out
name='openai/whisper-tiny';rev=MODELS[name]['sha'];print('download Whisper',flush=True)
folder=snapshot_download(name,revision=rev,cache_dir=str(ROOT/'model_cache'),allow_patterns=['*.json','*.txt',('model.safetensors' if 'model.safetensors' in MODELS[name]['files'] else 'pytorch_model.bin')])
# File is a public test recording distributed by the original implementation.
audio_url='https://raw.githubusercontent.com/openai/whisper/v20240930/tests/jfk.flac'
path=OUT/'jfk.flac';urllib.request.urlretrieve(audio_url,path)
audio,sr=sf.read(path);original_sr=sr
if audio.ndim>1:audio=audio.mean(axis=1)
if sr!=16000:
 import math
 g=math.gcd(sr,16000);audio=resample_poly(audio,16000//g,sr//g);sr=16000
processor=WhisperProcessor.from_pretrained(folder,local_files_only=True);model=WhisperForConditionalGeneration.from_pretrained(folder,local_files_only=True).eval()
encoded=processor(audio,sampling_rate=sr,return_tensors='pt',return_attention_mask=True)
features=encoded.input_features
model.generation_config.forced_decoder_ids=None
with torch.inference_mode():
 start=time.perf_counter();ids=model.generate(features,attention_mask=encoded.attention_mask,language='en',task='transcribe',do_sample=False,max_new_tokens=128);elapsed=time.perf_counter()-start
hyp=processor.batch_decode(ids,skip_special_tokens=True)[0]
ref='And so my fellow Americans ask not what your country can do for you ask what you can do for your country'
def words(s):return re.findall(r"[a-z]+(?:'[a-z]+)?",s.lower())
a,b=words(ref),words(hyp);dist=np.zeros((len(a)+1,len(b)+1),int);dist[:,0]=np.arange(len(a)+1);dist[0]=np.arange(len(b)+1)
for i in range(1,len(a)+1):
 for j in range(1,len(b)+1):dist[i,j]=min(dist[i-1,j]+1,dist[i,j-1]+1,dist[i-1,j-1]+(a[i-1]!=b[j-1]))
result=dict(model=name,revision=rev,audio_url=audio_url,audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original_sample_rate=original_sr,used_sample_rate=sr,duration_seconds=len(audio)/sr,reference=ref,hypothesis=hyp,normalization='lowercase, alphabetic word tokens, punctuation ignored',reference_words=len(a),edit_distance=int(dist[-1,-1]),wer=float(dist[-1,-1]/len(a)),generation_seconds=elapsed,rtf=elapsed/(len(audio)/sr),scope='one public English recording; not a corpus benchmark or a Russian ASR evaluation',meta=meta)
save('whisper_asr.json',result);print('Whisper',result,flush=True)
