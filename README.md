# ChartSketcher: Reasoning with Multimodal Feedback and Reflection for Chart Understanding

[[arXiv](https://arxiv.org/abs/2505.19076)]
[[ModelScope](https://www.modelscope.cn/models/HUANGMUYE/ChartSketcher-72B)]

A multimodal feedback-driven step-by-step reasoning method for chart understanding, built on Qwen2VL-72B with 72B parameters.

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

**Use ms-swift framework for training:**

```bash
# Install ms-swift
pip install ms-swift

# Training data is being organized and will be uploaded soon
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

## 📊 Status

- ✅ Model Released
- ✅ Inference Code Available  
- 🔄 Training Data (Coming Soon)

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
