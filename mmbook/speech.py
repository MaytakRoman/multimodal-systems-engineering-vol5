"""Глава 5: WER, DER, VAD и фиксация транскрипта перед действием."""
import itertools
import wave
import numpy as np
from .common import write_json


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current=[i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def wer(reference, hypothesis):
    a, b = reference.split(), hypothesis.split()
    if not a: return 0.0 if not b else None
    return edit_distance(a, b)/len(a)


def frame_der(reference, hypothesis):
    """Односпикерные кадры без collar/overlap; None означает тишину.
    Имена спикеров выравниваются глобально по минимальной ошибке.
    """
    if len(reference)!=len(hypothesis): raise ValueError('Различная сетка кадров')
    ref_ids=sorted(set(reference)-{None}); hyp_ids=sorted(set(hypothesis)-{None})
    if max(len(ref_ids),len(hyp_ids))>7: raise ValueError('Учебный точный перебор ограничен 7 спикерами')
    padded=ref_ids+[f'__unmatched_{i}' for i in range(max(0,len(hyp_ids)-len(ref_ids)))]
    if not hyp_ids: mappings=[{}]
    else: mappings=[dict(zip(hyp_ids,p)) for p in itertools.permutations(padded,len(hyp_ids))]
    def counts(mapping):
        missed=fa=confused=0
        for a,b in zip(reference,hypothesis):
            missed+=a is not None and b is None
            fa+=a is None and b is not None
            confused+=a is not None and b is not None and a!=mapping[b]
        return missed,fa,confused
    best=min((counts(m) for m in mappings), key=sum)
    speech=sum(x is not None for x in reference)
    return dict(miss=best[0],false_alarm=best[1],confusion=best[2],reference_speech_frames=speech,
                der=sum(best)/speech if speech else (0.0 if sum(best)==0 else None))


def energy_vad(samples, frame=320, threshold=0.04):
    a=np.asarray(samples,float)
    if frame<=0 or a.ndim!=1: raise ValueError('Неверный формат аудио')
    a=a[:len(a)//frame*frame].reshape(-1,frame)
    return np.sqrt((a*a).mean(axis=1))>=threshold


class TranscriptGate:
    """Ревизии одного turn; запускается только действие по final, однократно."""
    def __init__(self):
        self.revisions={}; self.texts={}; self.committed=set()

    def apply(self, event):
        turn=event['turn']; revision=event['revision']
        if turn in self.committed: return dict(status='already_committed')
        old=self.revisions.get(turn,-1)
        if revision<old: return dict(status='stale')
        if revision==old:
            if event['text']!=self.texts[turn]: return dict(status='revision_conflict')
            if not event['final']: return dict(status='duplicate')
        self.revisions[turn]=revision
        self.texts[turn]=event['text']
        if not event['final']: return dict(status='provisional')
        self.committed.add(turn)
        return dict(status='committed',text=event['text'],turn=turn,revision=revision)


def save_wave(path, audio, rate):
    if rate<=0: raise ValueError('Неверная частота')
    pcm=(np.clip(audio,-1,1)*32767).astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate); f.writeframes(pcm.tobytes())


def experiment(out, seed):
    """Создать тестовый WAV, оценить VAD/транскрипт и подавить устаревшие действия."""
    rate=16000; t=np.arange(rate*3)/rate
    active=((t>=.4)&(t<1.3))|((t>=1.7)&(t<2.7))
    audio=.2*np.sin(2*np.pi*220*t)*active
    save_wave(out/'synthetic_tone.wav',audio,rate)
    events=[dict(turn=1,revision=0,final=False,text='открой насос 7'),
            dict(turn=1,revision=1,final=False,text='открой паспорт насоса 17'),
            dict(turn=1,revision=0,final=True,text='открой насос 7'),
            dict(turn=1,revision=2,final=True,text='открой паспорт насоса 17'),
            dict(turn=1,revision=3,final=True,text='открой паспорт насоса 17')]
    gate=TranscriptGate(); actions=[gate.apply(e) for e in events]
    write_json(out/'stream_events.json',dict(events=events,actions=actions))
    ref=[None,'a','a','b','b',None]; hyp=[None,'y','y','x',None,'x']
    return dict(wer=wer('открой паспорт насоса семнадцать','открой паспорт насоса семь'),
                diarization=frame_der(ref,hyp),vad_speech_frames=int(energy_vad(audio).sum()),
                sample_rate=rate,duration_seconds=len(audio)/rate,actions=actions,
                limitation='WAV contains tones, not speech. No ASR, TTS or voice-cloning model is executed.')
