# Results

Held-out test set: 50 episodes (38 TalkRL + 12 Changelog), frozen before any data generation. Every system runs the same map-reduce pipeline (chunk notes → merged notes → article + social pack) with the same prompts and sampling. Numbers come from `results.json` in the runs listed at the bottom.

**Headline:** A QLoRA-fine-tuned 4B model (SFT + DPO), trained on Kaggle's free T4 GPUs, writes articles that are **10% less** faithful to the transcript than the base model (43.5% → 39.1% of sentences supported), with **19% fewer** invented entities per article, and an SEO checklist score of **96.5%** (base 80.5%), at 207 T4-seconds per episode, about **3× cheaper** per article than `qwen/qwen3.8-max-0902` via API (estimate).

## Main comparison

| System | Faithfulness ↑ | Invented entities / article ↓ | Invented numbers / article ↓ | Articles with any invention ↓ | SEO score ↑ | Flesch (target 50–70) | Valid social JSON ↑ | X posts ≤ 280 ↑ | Tokens/s | s / episode |
|---|---|---|---|---|---|---|---|---|---|---|
| Base (zero-shot) | 43.5% | 0.52 | 0.08 | 44.0% | 80.5% | 48.7 (50.0% in range) | 98.0% | 98.0% | 143 | 93 |
| SFT | 38.6% | 0.42 | 0.04 | 26.0% | 97.8% | 29.7 (2.0% in range) | 100.0% | 94.0% | 92 | 104 |
| SFT + DPO | 39.1% | 0.42 | 0.06 | 28.0% | 96.5% | 28.4 (0.0% in range) | 100.0% | 100.0% | 60 | 24 |
| SFT on base's notes | 37.1% | 0.44 | 0.02 | 26.0% | 91.0% | 25.0 (0.0% in range) | 100.0% | 90.0% | 59 | 25 |

Generation ran on 2 T4s (tensor parallel); s / episode is wall time.
SFT + DPO writes its articles from SFT's Stage A notes (DPO trained on Stage B only), so its tokens/s and s / episode cover Stage B only.
SFT on base's notes writes its articles from Base (zero-shot)'s Stage A notes (an ablation: does SFT lose faithfulness in its notes or in its articles?), so its tokens/s and s / episode cover Stage B only.

## Cost per article (estimate; no API calls)

Same pipeline, same prompts: 23,470 prompt + 9,535 completion tokens per episode (measured for SFT, whose notes SFT + DPO reuses), priced at published per-token rates (OpenRouter, checked 2026-10-21). Self-hosted = 207 T4-seconds per episode (wall time × T4s used) × one T4's hourly price. An API model would emit somewhat different token counts, so treat these as estimates.

| Option | USD per article | USD per 1,000 articles |
|---|---|---|
| Self-hosted fine-tuned 4B · kaggle_t4_free_tier | $0.0000 | $0.00 |
| Self-hosted fine-tuned 4B · aws_g4dn_xlarge | $0.0303 | $30.30 |
| API · `deepseek/deepseek-v4-pro` | $0.0173 | $17.31 |
| API · `qwen/qwen3.7-plus` | $0.0197 | $19.71 |
| API · `google/gemini-3.8-flash` | $0.0587 | $58.69 |
| API · `qwen/qwen3.8-max-0902` | $0.1042 | $104.15 |
| API · `anthropic/claude-sonnet-5.5` | $0.1423 | $142.29 |
| API · `openai/gpt-6.1-sol` | $0.1423 | $142.29 |

One-time cost of building the model: 28.28 GPU-hours (free on Kaggle; $14.88 at the reference GPU price).

## SEO checklist pass rates

| Check | Base (zero-shot) | SFT | SFT + DPO | SFT on base's notes |
|---|---|---|---|---|
| `kw_in_title` | 98.0% | 100.0% | 100.0% | 100.0% |
| `kw_in_first_100_words` | 96.0% | 98.0% | 98.0% | 94.0% |
| `kw_in_h2` | 70.0% | 96.0% | 96.0% | 80.0% |
| `title_len_ok` | 78.0% | 98.0% | 94.0% | 90.0% |
| `meta_len_ok` | 100.0% | 100.0% | 98.0% | 98.0% |
| `min_3_h2` | 100.0% | 100.0% | 100.0% | 100.0% |
| `faq_present` | 40.0% | 100.0% | 100.0% | 100.0% |
| `density_ok` | 62.0% | 90.0% | 86.0% | 66.0% |

Other: article produced Base (zero-shot) 100.0%, SFT 100.0%, SFT + DPO 100.0%, SFT on base's notes 100.0%; invalid Stage A JSON Base (zero-shot) 11.4%, SFT 2.5%, SFT + DPO 2.5%, SFT on base's notes 11.4%.

## LLM judge (pairwise vs base, teacher model as judge)

| System | Win | Tie | Loss |
|---|---|---|---|
| SFT | 12.0% | 40.0% | 48.0% |
| SFT + DPO | 14.0% | 34.0% | 52.0% |
| SFT on base's notes | 6.0% | 48.0% | 46.0% |

Each pair is judged in both A/B orders; a system wins an episode only if it wins one order and does not lose the other.

## Side-by-side examples

###  Aleksandra Faust (`talkrl:aleksandra-faust`)

**Base (zero-shot)** · keyword *compositional tasks* · faithfulness 50.0% · SEO 87.5% · invented entities 1

> # Compositional Tasks in Reinforcement Learning Meta description: How compositional tasks enable smarter, scalable AI decision-making in real-world environments. Reinforcement learning (RL) is evolving beyond simple games and simulations. Researchers are now focusing on **compositional tasks**—complex actions like booking airline tickets or navigating websites—where skills build on each other. Dr. Aleksandra Faust explains how AI agents can learn basic manipulation and navigation skills that combine into larger tasks. These tasks aren’t random. They follow patterns, like a chain of steps: open a menu, click a button, enter a password. This structure allows agents to generalize across environments. Instead of retraining for every new site, one agent can handle many tasks. The key is understanding how these tasks are built and how rewards shape learning. ## What Are Compositional Tasks? ### How They Work in Real Life Compositional tasks involve combining simple actions into complex ones. For example, buying a movie ticket requires selecting a show, choosing a time, and entering a seat. Each step is a small skill. The agent learns these individually, then links them together. This mirrors how humans learn: we don’t master all tasks at once. We build skills step by step. ### Why They Matter for AI These tasks are scalable. A single RL agent can complete many different ones without extra training. This is a …

**SFT** · keyword *reinforcement learning* · faithfulness 46.2% · SEO 100.0% · invented entities 1

> # Reinforcement Learning and Its Real-World Impact Meta description: Explore how reinforcement learning is shaping real-world applications and advancing AI research. Reinforcement learning (RL) is a powerful approach in artificial intelligence that enables systems to learn optimal behaviors through trial and error. In this article, we delve into the work of Dr. Aleksandra Faust, a Google Brain Research scientist, who is at the forefront of advancing RL for real-world applications. Her research focuses on making RL scalable and suitable for complex decision-making, particularly in areas like drone delivery and navigation. Faust's work highlights the challenges and opportunities in evolving rewards, explainable AI, and compositional tasks, offering insights into how RL can contribute to social good and accessibility. ## The Role of Reinforcement Learning in Real-World Applications Reinforcement learning has shown significant promise in solving real-world problems, from autonomous navigation to drone delivery. Dr. Aleksandra Faust's research emphasizes the importance of making RL scalable and applicable to complex decision-making scenarios. For instance, her work on UAVs (unmanned aerial vehicles) represents a pioneering application of RL in this domain. Faust notes that RL can automate repetitive tasks, benefiting people with accessibility needs, and that the field is at a critical juncture where multidisciplinary approaches are essential. ### Challenges in Reward Design and Algorithm Coupling One of the key challenges in reinforcement learning …

**SFT + DPO** · keyword *reinforcement learning* · faithfulness 34.0% · SEO 100.0% · invented entities 0

> # Reinforcement Learning and Its Real-World Impact Meta description: Explore how reinforcement learning is transforming real-world applications and advancing AI research. Reinforcement learning (RL) is a powerful approach in artificial intelligence that enables systems to learn optimal behaviors through trial and error. In this article, we dive into the work of Dr. Aleksandra Faust, a Google Brain Research scientist, who is at the forefront of advancing RL for complex decision-making and real-world applications. Her research highlights the potential of RL in areas like drone delivery, navigation, and even social good. As RL continues to evolve, it's becoming increasingly clear that its impact will be felt across industries, from robotics to healthcare. ## The Role of Reinforcement Learning in Real-World Applications Reinforcement learning has shown remarkable potential in solving real-world problems, particularly in areas that require complex decision-making. Dr. Faust's work focuses on making RL scalable and applicable to real-world scenarios, such as drone delivery and autonomous navigation. These applications often involve navigating unpredictable environments, which is a challenge for traditional AI systems. RL excels in such environments because it allows agents to learn from experience and adapt to changing conditions. ### Drone Delivery and Navigation One of the most promising applications of RL is in drone delivery and navigation. Dr. Faust's research has explored how RL can be used to …

**SFT on base's notes** · keyword *compositional tasks* · faithfulness 51.3% · SEO 100.0% · invented entities 1

> # Compositional Tasks in Reinforcement Learning Meta description: Explore how compositional tasks are shaping the future of reinforcement learning and AI generalization. In the world of artificial intelligence, reinforcement learning (RL) has emerged as a powerful tool for enabling agents to make decisions in complex environments. A key focus in this field is the development of **compositional tasks**, which involve breaking down complex problems into simpler, interconnected components. These tasks are crucial for enabling AI systems to generalize across different environments and scenarios, much like how humans learn to navigate the world through a combination of skills and experiences. Dr. Aleksandra Faust, a leading researcher in this area, discusses how **compositional tasks** are being used to train RL agents to perform a wide range of tasks, from web navigation to robot manipulation, with minimal additional training. ## The Role of Compositional Tasks in AI Generalization Compositional tasks are a fundamental concept in the field of AI, particularly in reinforcement learning. These tasks involve a series of basic manipulation and navigation skills that are connected in a dependency graph, allowing agents to perform complex actions by combining simpler ones. For example, in web navigation, an agent might need to complete a series of tasks such as clicking on a button, filling out a form, and navigating to a new page. Each …

### Amy Zhang (`talkrl:amy-zhang`)

**Base (zero-shot)** · keyword *reinforcement learning generalization* · faithfulness 43.6% · SEO 75.0% · invented entities 0

> # Reinforcement Learning Generalization in Robotics and Healthcare Meta description: How reinforcement learning generalization improves robotics and healthcare through structured models and causal insights Reinforcement learning generalization is key to building smart agents that work in new, unseen situations. Dr. Amy Zhang’s research focuses on how agents can learn from limited experience and apply that knowledge to new environments—especially in robotics and healthcare. Unlike supervised learning, where data comes from many similar cases, RL agents interact with environments and must adapt based on rewards. This makes generalization harder. Her work explores structured models like block MDPs and invariant causal prediction to help agents recognize patterns and avoid false signals. These tools improve how agents learn from few examples and transfer knowledge across tasks. ## What Is Reinforcement Learning Generalization? Reinforcement learning generalization means an agent performs well in new, unseen environments after training in a few. ### Why It’s Harder Than Supervised Learning In supervised learning, data is often labeled and consistent. In RL, the environment changes, and rewards are sparse. Agents don’t see all possible states. This makes it hard to predict what happens in new situations. Dr. Zhang explains that standard MDPs don’t support generalization to new states. If an agent learns in one setting, it can’t reliably act in a new one. This is a core limitation. …

**SFT** · keyword *reinforcement learning* · faithfulness 30.8% · SEO 100.0% · invented entities 0

> # Reinforcement Learning and Generalization in AI Meta description: Explore reinforcement learning and generalization in AI with insights from Dr. Amy Zhang's research on block MDPs and zero-shot generalization. In the world of artificial intelligence, reinforcement learning (RL) stands out as a powerful approach to teaching machines how to make decisions in complex environments. Dr. Amy Zhang, a postdoctoral scholar at UC Berkeley and research scientist at Facebook AI Research, is at the forefront of this field, focusing on how RL can be improved through better generalization. Her work delves into the challenges of out-of-distribution generalization, where models must perform well on tasks they haven't seen before, and the role of structured environments in enabling this. This article explores her research on reinforcement learning, including the use of block MDPs, invariant causal prediction, and zero-shot generalization, offering insights into the future of AI. ## Reinforcement Learning and the Challenge of Generalization Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. Unlike supervised learning, where models are trained on labeled data, RL agents learn through trial and error, receiving rewards or penalties based on their actions. However, one of the biggest challenges in RL is generalization—ensuring that an agent can perform well in new, unseen environments. This is particularly difficult …

**SFT + DPO** · keyword *reinforcement learning* · faithfulness 32.5% · SEO 100.0% · invented entities 0

> # Reinforcement Learning and Generalization in AI Meta description: Explore how reinforcement learning and generalization are shaping AI research, with insights from Dr. Amy Zhang. Reinforcement learning (RL) is a powerful approach in AI that enables agents to learn optimal behaviors through interaction with their environment. Dr. Amy Zhang, a postdoctoral scholar at UC Berkeley and research scientist at Facebook AI Research, is at the forefront of this field, focusing on how RL can generalize to new situations. Her work explores the challenges of out-of-distribution generalization, where agents must perform well in environments they haven't seen before. This is especially critical in real-world applications like robotics and healthcare, where unpredictability is the norm. Dr. Zhang's research delves into concepts like block MDPs, invariant causal prediction, and compositional generalization, all of which are essential for building more robust and adaptable AI systems. ## The Role of Reinforcement Learning in Generalization Reinforcement learning has long been a cornerstone of AI research, but its ability to generalize remains a significant challenge. In traditional RL, agents learn by interacting with an environment, receiving rewards or penalties based on their actions. However, when faced with new or unseen environments, these agents often struggle to adapt. This is where the concept of generalization becomes crucial. Dr. Amy Zhang highlights the importance of structured environments, such as …

**SFT on base's notes** · keyword *reinforcement learning generalization* · faithfulness 30.6% · SEO 87.5% · invented entities 0

> # Reinforcement Learning Generalization in Robotics and Healthcare Meta description: Explore reinforcement learning generalization in robotics and healthcare with insights from Dr. Amy Zhang. Reinforcement learning generalization is a critical challenge in robotics and healthcare, where agents must adapt to new environments and tasks without extensive retraining. Dr. Amy Zhang's research focuses on this area, particularly in the context of block MDPs and invariant causal prediction. Her work highlights the importance of structured MDPs in enabling agents to generalize across tasks, even in high-dimensional observation spaces. By leveraging causal inference and compositional learning, her research aims to bridge the gap between theoretical guarantees and practical applications in real-world scenarios. ## Reinforcement Learning Generalization: A New Frontier Reinforcement learning generalization is a key area of research that seeks to enable agents to perform well in new, unseen environments. Unlike supervised learning, where data is typically drawn from a well-defined distribution, reinforcement learning environments are often less structured and more dynamic. This makes generalization more challenging, as agents must learn to navigate a broader range of scenarios without explicit training on each one. ### The Role of Block MDPs Block MDPs are a type of structured MDP that assumes observations are fully observable due to disjoint observation sets per state. This assumption simplifies the state space and allows for more effective generalization. …

### The Changelog #662 (`changelog:the-changelog-662`)

**Base (zero-shot)** · keyword *spec-driven development* · faithfulness 52.8% · SEO 100.0% · invented entities 0

> # Spec-Driven Development Is Changing How We Build Software Meta description: Learn how spec-driven development is reshaping software creation with AI agents and new tools like Kiro. Spec-driven development is shifting how engineers build software. Instead of typing code first, developers now write clear specs in natural language. These specs describe what a feature should do, not how to build it. Tools like Kiro support this new workflow by letting users define problems in Markdown or EARS format. The focus is on clarity, context, and vision — not raw code. This change reflects a broader shift: engineers are no longer just typists. They’re now architects who guide AI agents through design, validation, and execution. The result? More thoughtful features, better team alignment, and a more joyful development process. ## What Is Spec-Driven Development? Spec-driven development replaces direct coding with structured planning. Engineers break down problems into clear, written specs. These specs act as blueprints for AI agents to follow. The process starts with a natural language description — like “build a dashboard that shows user activity” — then expands into detailed design documents. ### How It Works in Practice Users write specs in Markdown or EARS format. Kiro interprets these and generates design proposals, task breakdowns, and even code tasks. Developers can refine the spec instead of editing code directly. …

**SFT** · keyword *kiro ux* · faithfulness 25.6% · SEO 87.5% · invented entities 0

> # Kiro UX: The Future of Software Development Meta description: Explore the Kiro UX and how it's transforming software development with AI and agent-based workflows. In the rapidly evolving world of software development, tools like Kiro are redefining how engineers approach their work. Kiro UX, a new development tool, combines a code editor with an AI agent to streamline the creation of specifications and designs. This shift marks a significant evolution in how software is built, moving away from traditional code typing and toward a more intuitive, agent-driven workflow. Kiro UX is designed to make code generation more efficient by leveraging AI to handle the bulk of the work, allowing developers to focus on higher-level tasks. ## The Rise of Kiro UX and Agent-Based Development Kiro UX is a product of AWS's experience with the Q CLI, an agent-based tool that has become the most heavily used agent inside Amazon. The Q CLI started as an open-source project called Fig, which was later acquired by AWS. This experience led to the creation of Kiro, a tool that formalizes the 'vibe coding' process into a more robust, team-friendly workflow. Kiro UX emphasizes natural language interactions and markdown-based specifications over traditional code editing, making it easier for developers to communicate their ideas and intentions. ### The Role of Specifications in Modern Software …

**SFT + DPO** · keyword *kiro ux* · faithfulness 34.0% · SEO 87.5% · invented entities 0

> # Kiro UX: The Future of Software Development Meta description: Explore the Kiro UX and how it's transforming software development with AI and agent-based workflows. In the rapidly evolving world of software development, tools like Kiro are redefining how engineers approach their work. Kiro UX, a new development tool, combines a code editor with an AI agent to streamline the creation of specifications and designs. This shift is part of a broader trend where software engineers are transitioning from being 'typists' to 'drivers' of agent behavior. Kiro UX emphasizes natural language interactions and markdown-based specifications, offering a more intuitive and efficient workflow compared to traditional code editing. ## The Rise of Kiro UX and Agent-Based Development Kiro UX is a product of AWS's experience with the Q CLI, an agent-based tool that has become the most heavily used agent inside Amazon. The Q CLI started as an open-source project called Fig, which was later acquired by AWS. This experience led to the development of Kiro, a tool that formalizes the 'vibe coding' process into a more robust, team-friendly workflow. Kiro UX focuses on spec authoring rather than code typing, leveraging AI to handle the bulk of the work. ### The Evolution of Q CLI and Kiro The journey from Q CLI to Kiro highlights the evolution of agent-based development. Q …

**SFT on base's notes** · keyword *spec-driven development* · faithfulness 21.6% · SEO 100.0% · invented entities 0

> # Spec-Driven Development in the Age of AI Agents Meta description: Explore how spec-driven development is transforming software creation with AI agents and tools like Kiro. In the rapidly evolving world of software development, the concept of **spec-driven development** is gaining traction as a new paradigm. This approach, where developers focus on creating detailed specifications before writing code, is being reshaped by the rise of AI agents. Tools like Kiro, developed by AWS, are at the forefront of this shift, emphasizing the importance of natural language interaction and agent-driven workflows over traditional code editing. This article delves into the world of **spec-driven development**, exploring how it's changing the way engineers think about software creation and how tools like Kiro are making this process more intuitive and efficient. ## The Rise of Spec-Driven Development ### What is Spec-Driven Development? Spec-driven development is a methodology where developers break down complex problems into smaller, manageable components using pseudocode, design documents, or other forms of specification. This approach allows engineers to think through the problem before writing any code, ensuring that the solution is well-structured and aligned with the desired outcome. In the context of AI agents, this means that the agent is given a clear set of instructions or requirements to follow, rather than being asked to generate code on the fly. ### …

## Runs

- Base (zero-shot): `05_eval-v4`
- SFT: `05_eval-v4`
- SFT + DPO: `05_eval-v4`
- SFT on base's notes: `05_eval-v4`
