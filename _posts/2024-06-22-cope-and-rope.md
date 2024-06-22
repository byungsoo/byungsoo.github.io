---
layout: post
title:  CoPE | LLMs make silly mistakes on contextual reasoning
comments: true
tags: [paper review]
---
<!-- More : https://github.com/daattali/beautiful-jekyll?tab=readme-ov-file#main-parameters -->

It is not a surprise anymore that the latest LLMs, such as GPT4 or Llama-2 70B, sometimes fail at very simple tasks. In the example below, both words "Alice" and "book" don't appear on the last sentence, but the answers from LLMs are obviously wrong.

![llm_prompt](/assets/img/cope_why_img1.png)

It might be due to the limitation of [relative positional encoding](https://arxiv.org/pdf/1803.02155), as pointed out from [the paper](https://arxiv.org/pdf/2405.18719?trk=public_post_comment-text). For LLM, understanding the conecpt of `sentence` and attend with different weights for different sentence is not trivial. One may think that the latent concept of `sentence` is something learnable with the token representing `period (.)`, but assinging sentence-level weights for the tokens consisting of each sentence is challenging because we do not know how many tokens the last sentence has.




## Solution : Attend on learned context-changes-counter

Maybe counting position based on sentence changes might be a slightly better version, and we can encode position with a counter triggering whenever period `.` appears on the text might be better than encoding based on the `word counter`.

As mentioned in the paper, it might be even better if positions are measured in a context dependent way rather than being a simple token count. This can be done by first deciding (**learning**) which tokens should be included when measuring distance using their context vectors. The paper uses sigmoid function for soft gating function to make it differentiable and to be trained through backpropagation.

$$ g_{ij} = \sigma ({q_i}^T k_j), \text{ }\text{ } j < i $$

Position to encode is now simply by adding (**counting**) the gate values.

$$ p_{ij} = \Sigma_{k=j}^{i} g_{ik} $$

One thing to note is that, due to the sigmoid function appeared on a gating function, the position $p_{ij}$ can take non-integer values, and interpolation is recommended from [the paper](https://arxiv.org/pdf/2405.18719?trk=public_post_comment-text).

![llm_prompt](/assets/img/cope_how_img1.png)

### Computation
The most expensive computation is matrix multiplication of \\({q_i}^T k_j\\). However, note that this has been already calculated for regular attention block, and we can simply reuse it and avoid most of computational burden, except applying \\(\sigma(\cdot)\\). Another expensive part is when calculating \\(q_i^T e[p_{ij}]\\) as a part of \\(a_{ij} = \text{Softmax}(q_i^T (k_j + e[p_{ij}]))\\), and the paper suggest to limit the maximum possible position by setting \\(p_{ij} = \text{min}(p_{ij}, p_{max})\\), and set \\(p_{max}\\) reasonably small.

## Takeaways
* Context-level (not necessarily sentence) relative position can be encoded with CoPE, and there would be various type of high-level tasks, like counting, detecting, coding, and many more.
* Meaningful application and its training dataset is needed. The paper was using [Flip-Flop Task](https://arxiv.org/pdf/2306.00946), [Selective Copy Task](https://arxiv.org/pdf/2312.00752), as well as other LLM dataset and coding dataset
* Open question to revisit #1 : Is there a benefit by using CoPE and traditional relative/absolute PE together? This is not clearly answered from the paper.
* Open question to revisit #2 : What shall we do for high-level context for non-LLM application, for example, Euclidean space? That would be an interesting research question.