
#!/bin/bash

# This script runs every time your Studio starts, from your home directory.


# List files under fast_load that need to load quickly on start (e.g. model checkpoints).
#
# ! fast_load
# .ollama/models/blobs/sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1
# .ollama/models/blobs/sha256:e8a35b5937a5e6d5c35d1f2a15f161e07eefe5e5bb0a3cdd42998ee79b057730
# .ollama/models/blobs/sha256:e6836092461ffbb2b06d001fce20697f62bfd759c284ee82b581ef53c55de36e
# .ollama/models/blobs/sha256:ed11eda7790d05b49395598a42b155812b17e263214292f7b87d15e14003d337
# .ollama/models/blobs/sha256:f9b1e3196ecfce03ac97468fa0b6d85941fea050c6666c57f5856a8baec6507d
# ./weights/sentence_transformers/BAAI_bge-large-en-v1.5/model.safetensors
# weights/sentence_transformers/BAAI_bge-reranker-large/model.safetensors
# weights/sentence_transformers/BAAI_bge-reranker-large/pytorch_model.bin
# weights/hub/models--BAAI--bge-reranker-large/blobs/9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e
# weights/hub/models--BAAI--bge-reranker-large/blobs/68171d1ff68b731a33d119708476692c094a466b
# weights/hub/models--BAAI--bge-reranker-large/blobs/059214673d9d6d2ee319411e2ffec8c024b816d5
# weights/hub/models--BAAI--bge-reranker-large/blobs/bb100d3a6586ee71aed778b7566ccf68dd498947
# weights/hub/models--BAAI--bge-reranker-large/blobs/c5ae4e262c60ed2fb507ec587358285377ec36ee0a2e6da029f0534272b06d36
# weights/hub/models--BAAI--bge-reranker-large/blobs/cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865



# Add your startup commands below.
#
# Example: streamlit run my_app.py
# Example: gradio my_app.py
curl https://ollama.ai/install.sh | sh

mkdir -p ~/log
ollama serve > ~/log/ollama.log 2> ~/log/ollama.err &
# preload the model
curl http://localhost:11434/api/generate -d '{"model": "llama3:instruct"}'

ollama pull llama3:instruct
ollama run llama3:instruct