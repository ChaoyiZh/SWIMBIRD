export LMUData=/project/siyuh/common/chaoyi/code/SWIMBIRD/datasets/VLMEval

CUDA_VISIBLE_DEVICES=0 torchrun  --master_port=29500 --nproc_per_node=1 run.py --data DynaMath WeMath MathVerse_MINI HRBench4K HRBench8K VStarBench MMStar RealWorldQA --model SwimBird-SFT-8B --judge your_judge_model --api-nproc 10 --verbose 
