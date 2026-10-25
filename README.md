# FineTuning SLM QLoRA JurisEval

**FineTuning SLM QLoRA JurisEval** is a Proof of Concept (PoC) focused on the end-to-end cycle of adaptation, specialization, and evaluation (*fine-tuning & evaluation*) of Small Language Models (SLMs) tailored for the *Brazilian legal domain*. The pipeline automates efficient parameter training via **QLoRA (4-bit NF4 with the PagedAdamW optimizer)** using the `TRL/PEFT` framework under accessible hardware constraints (such as T4 GPUs), targeting a 1.5B parameter SLM (`Qwen2.5-1.5B-Instruct) for complex tasks in **Criminal Law, Cyber Law, and Data Protection**. Following training, the script executes a **customized multidimensional benchmark (JurisEval)** inspired by the three levels of *LawBench* (Memorization, Comprehension, and Application), rigorously measuring semantic quality metrics (**ROUGE-L**), strict technical vocabulary retention (*Keyword Recall*), memory consumption (**VRAM in MB**), and inference latency.

## Technical Highlights:
* **4-bit Optimized Training:** A clean implementation of QLoRA using `BitsAndBytesConfig` and `prepare_model_for_kbit_training`, enabling fine-tuning on modest GPUs.
* **Alignment and Formatting via Chat Template:** Standardization using Qwen instruction tokens (`<|im_start|>`, `<|im_end|>`) to enforce domain-specific legal persona behavior.

**Multidimensional Legal Evaluation (`JurisEval`):**

* **Textual Quality:** Similarity validation using *ROUGE-L* scored against reference target responses.
* **Doctrinal/Legal Compliance:** Automated verification of mandatory technical term retrieval (*Keyword Recall* focusing on statutes, ADPFs, and legal doctrines).
* **Computational Efficiency:** Dynamic runtime monitoring of peak VRAM usage and per-inference latency.

### ! Disclaimer

This Proof of Concept (PoC) exhibits critical structural limitations due to its reliance on a single-example training dataset, resulting in severe overfitting and rote memorization that leads to degenerate, repetitive output loops and a sharp performance decline in comprehension tasks where *ROUGE-L* plummets to *0.2537*. Furthermore, the *JurisEval* evaluation is statistically insufficient because it covers only three test cases and relies solely on lexical and syntactic metrics (*ROUGE-L* and *Keyword Recall*) that fail to assess true logical coherence or legal validity, alongside runtime dependency conflicts between Python packages (`pyarrow` and `cudf`). To address these shortcomings and advance the project toward production readiness or advanced research, it is essential to **expand the dataset with thousands of diverse, real-world legal instructions**, implement repetition penalties during text generation, containerize the environment using Docker to ensure technical reproducibility, and **strengthen JurisEval by incorporating domain-expert human evaluations, an LLM-as-a-judge framework, and robust literature benchmarks like *LawBench* and *LexRAG***.

# References

Haitao Li, Yifan Chen, Hu YiRan, Qingyao Ai, Junjie Chen, Xiaoyu Yang, Jianhui Yang, Yueyue Wu, Zeyang Liu, and Yiqun Liu. 2025. LexRAG: Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal Consultation Conversation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25). Association for Computing Machinery, New York, NY, USA, 3606–3615. https://doi.org/10.1145/3726302.3730340

Mao, Y., Ge, Y., Fan, Y. et al. A survey on LoRA of large language models. Front. Comput. Sci. 19, 197605 (2025). https://doi.org/10.1007/s11704-024-40663-9

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, Jidong Ge, and Vincent Ng. 2024. LawBench: Benchmarking Legal Knowledge of Large Language Models (https://aclanthology.org/2024.emnlp-main.452/). In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7933–7962, Miami, Florida, USA. Association for Computational Linguistics.
