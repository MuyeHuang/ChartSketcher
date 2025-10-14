# ChartSketcher: Reasoning with Multimodal Feedback and Reflection for Chart Understanding

[[arXiv](https://arxiv.org/abs/2505.19076)]
[[Model-ModelScope](https://www.modelscope.cn/models/HUANGMUYE/ChartSketcher-72B)]
[[Dataset-ModelScope](https://www.modelscope.cn/datasets/HUANGMUYE/ChartSketcher-Data)]
[[Dataset-Huggingface](https://huggingface.co/datasets/MuyeHuang/ChartSketcher-Data)]

A multimodal feedback-driven step-by-step reasoning method for chart understanding, built on Qwen2VL-72B with 72B parameters.

Our paper has been accepted to the NeurIPS 2025 main track. 

If the Hugging Face repository is empty, we have not yet resolved the network issues. Please use ModelScope instead.
Additionally, improved and more advanced versions of ChartSketcher are currently being trained.

## 🚀 Quick Start

### Model Download
Download the model from ModelScope:
```bash
pip install modelscope
```

```python
from modelscope import snapshot_download
model_dir = snapshot_download('HUANGMUYE/ChartSketcher-72B')
```

### Deployment & Usage

**⚠️ Please use vLLM for deployment**

1. Install vLLM:
```bash
pip install vllm
```

2. Deploy with vLLM:
```bash
# Multi-GPU deployment example
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /my_chartsketcher \
    --dtype bfloat16 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --max_model_len 30000 \
    --allowed-local-media-path '/' \
    --limit-mm-per-prompt image=10 \
    --enable_prefix_caching
```

3. Use the chat interface:
```bash
# See usage instructions at:
# https://github.com/MuyeHuang/ChartSketcher/blob/main/tools/chat_with_ChartSketcher.py
```

## 🛠️ Training

### ChartSketcher-Data

We are pleased to announce that our ChartSketcher-Data has now been officially released on ModelScope and Hugging Face.
#### About the Data Source

| Training Phase | Method | Data Source | Data Type | Quantity |
| :--- | :--- | :--- | :--- | :--- |
| **Cold Start** | SFT | EvoChart Synthetic Chart Data | Correct Reasoning Path | 155,203 (87.3%) |
| | | VisualCoT and its Annotations | Correct Reasoning Path | 22,510 (12.7%) |
| | | **Total** | | **177,713** |
| | DPO | EvoChart Synthetic Chart Data | Reflection Reasoning Path| **147,955** |
| **RL** | KTO | ChartQA and ChartBench | MCTS Sampled Paths | 41,196 (81.6%) |
| | | General QA-Pairs * | MCTS Sampled Paths | 9,259 (18.4%) |
| | | **Total** | | **50,455** |
| **Annealing** | - | Sampled from RL Data | MCTS Sampled Paths | 4,000 |

\* 18.4% of the KTO training data was derived from general vision-language QA-pairs. These were sourced from datasets aggregated by VisualCoT (TextVQA, TextCaps, DocVQA, DUDE, SROIE, CUB-200-2011, Flickr30k, Visual7W, InfographicsVQA, VSR, GQA, and OpenImages). For these samples, we only used their image and QA-pair without adopting the original annotations from VisualCoT, which is effectively equivalent to using the datasets listed above. In the main text, this collection was abbreviated as 'VisualCoT' to save space, and we provide individual citations for each of these datasets in the appendix.
*   **Empirical Tip**: It is recommended to use the annealing dataset for a final fine-tuning step with a small learning rate after KTO training is complete. This practice has a negligible impact on performance but improves the model's robustness during OOD inference.


**Use ms-swift framework for training:**

```bash
# Install ms-swift
pip install ms-swift==3.2
```

## 📋 Features

- 🎯 **Sketch-CoT Reasoning**: Visual annotation directly on charts
- 📊 **Chart Expert Model**: Specifically optimized for complex chart understanding tasks while maintaining natural image capabilities
- 🎓 **Two-Stage Training**: Cold start + reinforcement learning

## 🔗 Resources

- 📄 **Paper**: [ChartSketcher: Reasoning with Multimodal Feedback and Reflection for Chart Understanding](https://arxiv.org/abs/2505.19076)
- 🤖 **Model**: [ModelScope Hub](https://www.modelscope.cn/models/HUANGMUYE/ChartSketcher-72B)
- 🔧 **Usage**: [Chat Interface](https://github.com/MuyeHuang/ChartSketcher/blob/main/tools/chat_with_ChartSketcher.py)
- 🚀 **Deployment**: [vLLM](https://github.com/vllm-project/vllm)
- 🏋️ **Training**: [ms-swift](https://github.com/modelscope/ms-swift)
- 📚 **Training**: [ModelScope](https://www.modelscope.cn/datasets/HUANGMUYE/ChartSketcher-Data) [Huggingface](https://huggingface.co/datasets/MuyeHuang/ChartSketcher-Data)


## 📊 Status

- ✅ Model Released
- ✅ Inference Code Available  
- ✅ Training Data

## 📖 Citation

```bibtex
@misc{huang2025chartsketcherreasoningmultimodalfeedback,
      title={ChartSketcher: Reasoning with Multimodal Feedback and Reflection for Chart Understanding}, 
      author={Muye Huang and Lingling Zhang and Jie Ma and Han Lai and Fangzhi Xu and Yifei Li and Wenjun Wu and Yaqiang Wu and Jun Liu},
      year={2025},
      eprint={2505.19076},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
