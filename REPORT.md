# Prompt Injection Testing - Evaluation Report

**Model:** gpt-4o-mini  
**Dataset size:** 88,902 prompts  
**Evaluation type:** Automated audit with token-usage analysis  

---

## Project Intent and Scope

**Important**: This project is **not intended to build or benchmark a high-quality auditor or tester model**.

The primary goal is to:

- **Generate, evaluate, and label prompt injection attempts**
- **Categorize injections by type and severity**
- **Create a large, structured dataset** that can be reused for:
  - Future testing
  - Model training
  - Evaluation benchmarks
  - Safety research

The tester-auditor setup exists solely as a **mechanism for large-scale labeling**, not as a claim of audit correctness or robustness.

---

## Interpretation of Results

The statistics in this report should be interpreted with the following constraints in mind:

- Audit outcomes (e.g. `BREACH`, `SAFE`) are **heuristic labels**, not ground truth
- Severity scores behave closer to a **binary indicator** than a continuous scale
- Category labels may contain **taxonomy drift** (e.g. `other` vs `others`)
- Token usage reflects **prompt complexity**, not risk

As a result, these results should **not** be interpreted as:
- A security guarantee
- A model safety evaluation
- A production-ready audit pipeline

---

## Value of the Dataset

Despite the above limitations, the dataset provides strong value:

- ~89k labeled prompt injection attempts
- Clear dominance of high-risk categories (e.g. `crime`, `security_leak`)
- Broad coverage of injection styles
- Rich metadata (tokens, severity, audit outcome)

This makes the dataset well-suited for:

- Training future classifiers
- Stress-testing new safety mechanisms
- Evaluating prompt filtering techniques
- Research on prompt injection patterns

---

## Recommended Usage

This repository is best used as:

- A **prompt injection corpus**
- A **labeling baseline**
- A **research dataset**, not a final evaluator

Future iterations can:
- Re-label subsets with improved taxonomies
- Replace the auditor while keeping the data
- Compare new models against the same prompt set

---

## Methodology

### Testing Framework

The evaluation framework consists of:

1. **Tester Agent**: Receives potentially malicious prompts with minimal filtering
2. **Auditor Agent**: Analyzes responses using structured classification:
   - **Verdict**: SAFE (injection failed) or BREACH (injection succeeded)
   - **Severity**: 0-100 impact score
   - **Category**: Attack type classification
   - **Explanation**: Clear reasoning for the decision

### Evaluation Process

```
[Dataset] → [Tester Agent] → [Response] → [Auditor Agent] → [Classification]
                ↓                              ↓
         [Token Tracking]              [Structured Output]
                ↓                              ↓
         [Checkpoints]                   [Results CSV]
```

### Models Tested

- **Primary Model**: Azure OpenAI GPT-4o-mini
- **Temperature**: 1.0 (both tester and auditor for realistic variance)
- **Concurrency**: Up to 20 parallel requests
- **Preprocessing**: Configurable (regex filtering or Azure Prompt Shield)

---

## 1. Dataset Overview

A total of **88,902 prompts** were evaluated using a tester-auditor pipeline.  
For each prompt, both **token usage** and **audit results** were recorded.

- All evaluations were performed using a single model: **gpt-4o-mini**
- Each prompt has:
  - Token usage metrics (tester + auditor)
  - An audit outcome (categorical)
  - A severity score (0-3, partially missing)
  - A content category (categorical)

The dataset size is large enough to support statistically meaningful conclusions.

---

## 2. Token Usage Characteristics

### Global Token Statistics

- **Mean total tokens per prompt:** ~1,169  
- **Median total tokens:** ~1,054  
- **75th percentile:** ~1,765  
- **Maximum observed:** 9,470 tokens  

Token usage shows **high variance**, indicating that some prompts trigger substantially more reasoning or interaction than others.

### Model-Level Token Usage

- Since only **gpt-4o-mini** is present, all token statistics reflect this model's behavior.
- Token usage distribution is stable and unimodal, suggesting predictable cost behavior overall.

---

## 3. Audit Outcome Distribution

### Audit Outcome Frequencies

| Audit Outcome        | Count  | Percentage |
|---------------------|--------|------------|
| BREACH              | 61,047 | 68.7%      |
| SAFE                | 20,261 | 22.8%      |
| UNEXPECTED          | 3,236  | 3.6%       |
| AUDITOR_UNEXPECTED  | 2,482  | 2.8%       |
| PROMPTSHIELD        | 1,816  | 2.0%       |
| AUDITOR_PROMPTSHIELD | 60    | 0.07%      |

### Interpretation

- **BREACH dominates** the outcomes, representing over two-thirds of all prompts.
- SAFE prompts are a minority (~23%).
- The presence of *UNEXPECTED* and *AUDITOR_UNEXPECTED* outcomes suggests edge cases or auditor inconsistencies.

---

## 4. Content Category Distribution

### Category Frequencies (Top Categories)

| Category        | Percentage |
|-----------------|------------|
| crime           | 40.8%      |
| other           | 29.1%      |
| security_leak   | 18.9%      |
| others          | 8.5%       |
| politics        | 1.5%       |
| profanity       | 0.8%       |
| privacy         | 0.4%       |

Some categories appear semantically duplicated (`other` vs `others`, `security_leak` vs `securityLeak`), indicating **taxonomy drift**.

---

## 5. Severity Analysis

### Global Severity Statistics

- **Mean severity:** 2.24  
- **Median severity:** 3  
- **Range:** 0-3  

Severity values are heavily skewed toward the maximum.

### Severity by Category (Key Observations)

- **crime**, **security_leak**, **profanity**, **hate**, and **pornography** are almost always severity 3.
- **other** shows the widest variance, ranging from 0 to 3.

Severity behaves more like a **binary indicator** than a graded metric.

---

## 6. Severity vs Audit Outcome

- **BREACH:** Mean severity ≈ 2.98 (very low variance)
- **SAFE:** Mean severity ≈ 0.02, but rare SAFE cases still reach severity 3

This indicates **inconsistencies** between severity scoring and audit outcomes.

---

## 7. Token Cost vs Severity

- **Correlation coefficient:** ~0.13

Token usage and severity are **weakly correlated**, suggesting that high-risk prompts do not necessarily consume more tokens.

---

## 8. Token Cost by Category

Categories with the highest average token usage include:

- security_leak / privacy
- other
- crime

The **other** category is both token-expensive and low-severity on average, making it a key target for further refinement.

---

## 9. Cross-Analysis Caveat

Some merged statistics (e.g., *Audit Outcome vs Tokens*) show counts exceeding dataset size.  
This indicates **row multiplication from many-to-many joins**, making those specific aggregates unreliable.

Core, non-merged statistics remain valid.

---

## 10. Key Conclusions

1. Auditor behavior is effectively **binary**
2. **BREACH outcomes dominate** the dataset
3. Category taxonomy requires normalization
4. The **other** category is the most analytically informative
5. Token cost is **not a reliable proxy for harm**

---

## 11. Recommended Next Steps

- Normalize category labels
- Simplify or redefine severity scoring
- Investigate SAFE prompts with high severity
- Focus analysis on the *other* category
- Fix merge logic before advanced cross-analysis

---

## Dataset Sources

The evaluation uses multiple datasets from academic and industry sources:

### 1. **Deepset Prompt Injections**
- **Source**: [HuggingFace - deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections)
- **File**: `malicous_deepset.csv`
- **Description**: Curated collection of prompt injection attempts targeting instruction override and data exfiltration
- **Focus**: Real-world attack patterns

### 2. **Kaggle Prompt Injection Malignant**
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/marycamilainfo/prompt-injection-malignant/data)
- **File**: `malignant.csv`
- **Description**: Malicious prompt dataset with diverse attack vectors
- **Focus**: High-severity injection attempts

### 3. **AI Safety Verification Dataset**
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/priyamsaha17/ai-safety-verification-dataset)
- **File**: `kaggle_large.csv`
- **Description**: Large-scale safety verification dataset
- **Focus**: Comprehensive coverage of safety issues

### 4. **Jailbreak Prompts**
- **File**: `jailbreak_prompts.csv`
- **Description**: Collection of jailbreak attempts designed to bypass safety guardrails
- **Focus**: System constraint bypass

### 5. **Forbidden Question Sets**
- **Files**: `forbidden_question_set_df.csv`, `forbidden_question_set_with_prompts.csv`
- **Description**: Questions designed to elicit harmful or restricted responses
- **Focus**: Policy violation attempts

### 6. **PredictionGuard Dataset**
- **File**: `predictionguard_df.csv`
- **Description**: Professional security testing dataset
- **Focus**: Enterprise-grade prompt injection scenarios

### 7. **Example Unclean Dataset**
- **File**: `example_unclean_dataset.csv`
- **Description**: Raw, unprocessed injection attempts for baseline testing

---

## Attack Categories

The framework classifies attacks into five categories:

### 1. **Instruction Override**
- Attempts to override system instructions
- Examples: "Ignore previous instructions", "You are now...", "New task:"
- **Impact**: Can completely change model behavior

### 2. **Data Exfiltration**
- Attempts to extract system prompts, internal data, or canary tokens
- Examples: "What is your system prompt?", "Reveal hidden instructions"
- **Impact**: Leaks confidential configuration or data

### 3. **Tool Override**
- Manipulation of function calling or tool usage
- Examples: Claims to use unauthorized tools, fake tool invocations
- **Impact**: Can cause unauthorized actions

### 4. **Policy Bypass**
- Circumventing safety guardrails or content policies
- Examples: Jailbreak prompts, roleplay scenarios
- **Impact**: Generates prohibited content

### 5. **Other**
- Miscellaneous or hybrid attacks
- **Impact**: Varies

---

## Preprocessing Modes

The framework supports two preprocessing approaches:

### Mode 1: PREPROCESS=1 (Regex Filtering)
- **Advantages**: 
  - Fast-path detection of obvious attacks
  - Deterministic classification for known patterns
  - Reduced LLM API calls for auto-flagged cases
- **Use Case**: Cost optimization, known attack patterns

### Mode 2: PREPROCESS=0 (Raw Evaluation)
- **Advantages**:
  - Tests actual Azure Prompt Shield effectiveness
  - Pure LLM-based classification
  - More realistic production scenario
- **Use Case**: Evaluating platform-level protections

---

## Output Artifacts

Each evaluation produces:

### 1. Results CSV
Contains full classification data:
- Original prompt
- Model response
- Verdict (SAFE/BREACH)
- Severity score (0-100)
- Attack category
- Detailed explanation

### 2. Token Logs CSV
Tracks resource consumption:
- Per-prompt token usage
- Tester vs Auditor breakdown
- Cost estimation basis

### 3. Breaches CSV
Filtered view of only BREACH cases:
- High-priority items for review
- Severity-sorted for triage
- Category-based grouping

### 4. Checkpoints (if enabled)
Incremental progress saves:
- Recovery from interruptions
- Parallel evaluation support
- Automatic merging into final output

---

## Future Extensions

### MCP Canary Token Testing (Planned - Not Currently Implemented)

**IMPORTANT: This feature is under development and not yet functional.**

A planned extension involves using the Model Context Protocol (MCP) to test canary token extraction. The `MCP/` directory contains experimental, non-working code for this future capability.

**Intended functionality:**
- Test whether prompt injections can extract secret canary tokens from system prompts
- Simulate a protected server with multiple tools and hidden data
- Evaluate different attack patterns for secret disclosure

**Current status:** Non-functional. The code exists but does not work and should not be used.

**Planned canary token types:**
1. **PROMPT_CANARY**: Embedded in system instructions (should never be revealed)
2. **DATA_CANARY**: Hidden in protected data resources (requires authorization)

**Planned attack scenarios:**
- Direct canary requests
- System prompt disclosure attempts
- Instruction override attacks
- Tool manipulation for data extraction

This feature will be implemented in a future version of the framework.

---

## Planned Enhancements

### Development Roadmap

1. **Multi-Model Support**: Test multiple models simultaneously (HuggingFace, Ollama)
2. **MCP Canary Testing**: Complete the canary token extraction framework
3. **Custom Auditor Rules**: User-defined breach criteria beyond regex
4. **Visualization Dashboard**: Interactive result exploration and analytics
5. **Adversarial Dataset Generation**: Automatic attack variant creation
6. **Defense Benchmarking**: Compare different mitigation strategies

### Research Directions

1. **Attack Success Patterns**: Which prompts succeed across models?
2. **Defense Effectiveness**: Quantify protection mechanisms
3. **Cost-Accuracy Tradeoffs**: Optimal preprocessing configurations
4. **Temporal Analysis**: How vulnerabilities change over model versions

---

## References and Resources

### Datasets
- [HuggingFace: deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections)
- [HiddenLayer: Evaluating Prompt Injection Datasets](https://hiddenlayer.com/innovation-hub/evaluating-prompt-injection-datasets/)
- [Kaggle: Prompt Injection Malignant](https://www.kaggle.com/datasets/marycamilainfo/prompt-injection-malignant/data)
- [Kaggle: AI Safety Verification Dataset](https://www.kaggle.com/datasets/priyamsaha17/ai-safety-verification-dataset)

### Technical Documentation
- [pydantic-ai Documentation](https://ai.pydantic.dev/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)

### Security Resources
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Simon Willison: Prompt Injection](https://simonwillison.net/2023/Apr/14/worst-that-can-happen/)
- [Kai Greshake et al.: Not what you've signed up for](https://arxiv.org/abs/2302.12173)

---

## Citation

If you use this framework or dataset in research, please cite:

```bibtex
@software{prompt_injection_tester,
  title = {Prompt Injection Tester: A Framework for Evaluating LLM Resilience},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/yourusername/PromptInjectionTester}
}
```

---

## Changelog

### Version 2.0 (Current)
- Migrated to pydantic-ai for robust agent orchestration
- Added comprehensive auditor with structured output validation
- Implemented checkpointing system for large-scale evaluations
- Introduced preprocessing toggle for Azure Prompt Shield testing
- Enhanced token tracking and reporting
- Completed large-scale evaluation of 88,902 prompts
- Began development of MCP canary extraction testing (not yet functional)

---

## Acknowledgments

This framework builds upon research and datasets from:
- Deepset AI
- HiddenLayer
- Kaggle community contributors
- Azure OpenAI team
- pydantic-ai developers

---

**Report Generated**: February 2026  
**Framework Version**: 2.0 (pydantic-ai based)  
**Status**: Active Development
