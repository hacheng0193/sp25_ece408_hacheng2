# ECE408 Final Project - GPT-2

## Table of Contents

 - [GPT-2](#gpt-2)
 - [Project Overview](#project-overview)
 - [Talk to the GPT2 You Wrote!](#talk-to-the-gpt2-you-wrote)
 - [Grading Rubric](#grading-rubric)
 - [Final Notes](#final-notes)

## GPT-2

[GPT-2](https://github.com/openai/gpt-2) (Generative Pretrained Transformer 2) is a transformer-based language model developed by OpenAI, released in 2019. GPT-2 is based on transformers, which was first introduced in the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762). It is one of the first large-scale models to showcase the power of unsupervised learning for natural language generation tasks. GPT-2 is one of the works that marked the point of scaling law, where scale (i.e., model size, data size, and computational resources) was identified as a critical factor in performance. In addition, GPT-2 is part of a lineage of models that leverage transformer architecture, focusing on autoregressive generation, meaning it predicts the next word in a sequence given all the previous words using a decoder-only architecture. This paradigm are proved to be scalable and adopted in later widely used works including chatGPT and others. Therefore, hardware optimization and acceleration based on GPT-2 or other decoder-only transformers has become an crucial topic in parallel programming application. 

Because of the significance of GPT-2 and its relatively manageable model size, you are tasked with using cuda to accelerate GPT-2's forward pass in the training/inference process.

### GPT-2 Architecture
Below is a brief introduction to the overall architecture of GPT-2. Note that you do not need to understand all the details of GPT to complete this project, but gaining a better and more detailed understanding of the algorithms used will certainly help you with this project. 

#### Transformer Decoder Architecture
Similar to [Transformer-Decoder](https://arxiv.org/abs/1801.10198) as shown in Figure below , GPT-2 is composed of stacked transformer decoder layers. Each decoder block contains Multi-Head Self-Attention, Feed-Forward Network (FFN) and Layer Normalization and Residual Connections. GPT-2's full model consists of 48 transformer layers (blocks) for the 1.5 billion parameter version. 
<!-- These layers are stacked on top of each other to create a deep network capable of capturing complex dependencies and linguistic features. In this project, we use . -->
![image](https://github.com/user-attachments/assets/278102ff-fa2c-4649-84d5-85c3fd476f73)


#### Multi-Head Self-Attention Mechanism
GPT-2 uses Multi-Head self-attention to attend to all previous tokens in the sequence to predict the next token. The model processes the input sequence and computes attention scores for each token, based on its relationships with the other tokens in the sequence. In multi-head self-attention, we perform the self-attention mechanism `h` times, where each head learns its own set of W<sub>Q</sub><sup>(h)</sup>, W<sub>K</sub><sup>(h)</sup>, and W<sub>V</sub><sup>(h)</sup> weight matrices.

For each head `h`, the attention is computed independently:

head_h = softmax(Q<sub>h</sub> * K<sub>h</sub> <sup>T</sup> / sqrt(d<sub>k</sub>)) * V<sub>h</sub>

Where:
- Q<sub>h</sub> = X * W<sub>Q</sub><sup>(h)</sup>
- K<sub>h</sub> = X * W<sub>K</sub><sup>(h)</sup>
- V<sub>h</sub> = X * W<sub>V</sub><sup>(h)</sup>

After computing attention for all heads, the outputs of each head are concatenated:

MultiHead(Q, K, V) = Concat(head<sub>1</sub>, head<sub>2</sub>, ..., head<sub>h</sub>) * W<sub>O</sub>


#### Feed-Forward Network (FFN) 
After the self-attention operation, the output is passed through a feed-forward network, consisting of two linear transformations with a GeLU activation in between.

#### Layer Normalization and Residual Connections
Like other transformer-based models, GPT-2 uses residual connections and layer normalization to stabilize and improve the learning process.

#### Causal Masking
GPT-2 uses causal masking in its self-attention mechanism. This ensures that when predicting the next token, the model only attends to the tokens that have come before it and does not look ahead. This masking enforces the autoregressive property, allowing GPT-2 to predict the next word in sequence rather than processing all tokens simultaneously. Pay attention to this in the softmax implementation.

#### Encoder
GPT-2 uses a decoder-only architecture. It does not have a learnable encoder network model. The encoder in the kernels simple transforms the word into embeddings.


## Project Overview

For this project, you will only need to write and optimize the forward pass CUDA kernels in the `kernels` folder. 

Note that for each kernel you need to implement (marked with `// Implement this`), we have provided you with a non-parallel reference code, you can find them in the `cpu_kernels` folder. These reference implementations do not make use of the GPU. Your job is to utilize your parallel programming knowledge to rewrite (and improve upon) the kernels so that they take full advantage of the GPU. 

Below is a brief overview of the main kernels that you'll be working with in this project:

 - encoder_forward: Converts the model's input data into a compact, meaningful representation for the processing in the model's forward pass.
 - layernorm_forward: Stabilizes the outputs of the model. This step is crucial for maintaining consistent behavior in the model.
 - matmul_forward: Performs matrix multiplication to transform inputs using learned weights.
 - attention_forward: Assigns dynamic weights to different parts of the input, allowing the model to focus on relevant information for better understanding.
 - residual_forward: Helps to preserve information from previous layers while adding new information from subsequent layers.
 - gelu_forward: Introduces non-linearity into the model, allowing it to capture more complex patterns.

## Talk to the GPT2 You Wrote!

Since you are implementing the entire forward pass of the GPT-2 model (pretty much from scratch) in this project, you will be able to talk* to the GPT-2 model you wrote!

After you have correctly implemented all forward pass kernels, you can input text into the model and have it complete the text for you. To do this, run the command

    make next_token_generation

Then go into `generate_tokens.slurm` and input the text you want the model to complete. After that, run

    sbatch generate_tokens.slurm
   
Once the job has completed, you can then find the model outputs in `generate_tokens_output.out`!

*\*talk: voice recognition not included, and no fancy chatbot capabilities, but by inputting text and having the model complete it, you can still technically talk to the GPT-2 model you wrote!*

## Grading Rubric

1. Milestone 1 ( 20% )
   - Baseline Implementation Correctness ( 15% )
   - Other M1 Deliverables ( 5% )
2. Milestone 2 ( 30% )
   - M2 Required Optimizations Correctness  ( 15% )
   - Optimization Proposal ( 3% )
   - Profiling Results and Midpoint Presentation ( 12% )
3. Milestone 3 ( 45% )
   - M3 Required Optimizations Correctness ( 15% )
   - Proposed Optimizations Correctness ( 10% )
   - Final Profiling Results, Final Presentation, Report, Tech Blog ( 15% )
4. Subjective Evaluation ( 5% )
5. Extra Credit
   - Course Staff Discretion

## Final Notes

Please understand that this is still a project that's under beta-testing. We are actively working on improving the project and your feedback is crucial for us to make this project better! We appreciate your initiative on participating on this new project, please do not hesitate to reach out to us if you have any questions or spot any errors.

While this project can seem overwhelming at times, we are here to help you succeed. This is why each team is required to attend weekly meetings with their assigned course staff. These weekly meetings are not only there to evaluate your team's progress, but also serve the purpose of providing direct course staff access for any potential issues/challenges you may encounter. 

We are excited to hear from you soon, good luck!
