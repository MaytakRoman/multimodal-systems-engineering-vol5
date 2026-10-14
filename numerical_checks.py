"""Независимые арифметические проверки основных инженерных отношений."""
from pathlib import Path
import argparse
from mmbook.common import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results'))
    args=parser.parse_args()
    checks={
        'vision_tokens_224_patch16':(224//16)**2,
        'vision_tokens_448_patch16':(448//16)**2,
        'attention_ratio_double_resolution':((448//16)**4)/((224//16)**4),
        'mono_pcm_16khz_16bit_bytes_per_second':16000*2,
        'lora_4096_rank8_parameters':8*(4096+4096),
        'fp16_7b_weight_bytes':7_000_000_000*2,
        'int8_7b_weight_bytes_without_scales':7_000_000_000,
        'mask_iou_80_intersection_120_union':80/120,
        'context_remaining':8192-3100-8*256-1200,
    }
    expected=[196,784,16,32000,65536,14000000000,7000000000,2/3,1844]
    assert len(checks)==len(expected)
    assert all(abs(actual-target)<=1e-12 for actual,target in zip(checks.values(),expected))
    write_json(args.output/'numerical_checks.json',checks)
    print('9 arithmetic checks passed')


if __name__=='__main__': main()
