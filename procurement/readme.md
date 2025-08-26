
# Zimbabwean Procurement Internship Simulation

A **web-based interactive simulation** for a Zimbabwean procurement internship.  
Users take the role of a procurement intern and make decisions across multiple days. AI agents (Finance, Warehouse, Supplier, Logistics, Legal, Mentor) evaluate these decisions and provide real-time feedback.

---

## 🌟 Features

- Multi-agent procurement scenario simulation  
- Interactive **Streamlit UI**  
- AI-driven mentor and departmental feedback  
- Supports multiple suppliers, budgets, and compliance events  
- Real-time display of AI evaluations  

---

## 🛠 Technologies Used

- **Python 3.10+**  
- [Streamlit](https://streamlit.io/) for the web UI  
- [CrewAI](https://github.com/yourorg/crewai) for multi-agent orchestration  
- [LangChain](https://www.langchain.com/) with **ChatOllama** (local LLM)  
- Optionally supports OpenAI or other LLM backends  

---

## 💻 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/procurement-internship-sim.git
cd procurement-internship-sim
````

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup LLM (Local Ollama)

```bash
# Install Ollama
ollama pull mistral
```

> The default LLM is **Ollama LLaMA 3.2**. You can change to other providers (OpenAI, Google Gemma, TogetherAI) by editing `LLM` setup in `app.py`.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📝 Usage

1. Select a **Scenario/Day** from the sidebar.
2. Read the scenario details (warehouse request, budget, suppliers, special events).
3. Enter your decision in the **User Decision** text box (supplier, quantities, justification).
4. Click **Submit Decision**.
5. Watch AI agents evaluate your decision in real-time and provide feedback.

---

## ⚙️ Customization

* Extend `SCENARIOS` with more days, suppliers, and events.
* Adjust agent behavior by modifying `build_*_agent()` functions.
* Swap LLM backends by changing the `LLM` initialization.

---

## 🎨 Agents & Roles

| Agent     | Role                                                    |
| --------- | ------------------------------------------------------- |
| Mentor    | Senior Procurement Officer, evaluates intern decisions  |
| Finance   | Finance Officer, checks budget and cost savings         |
| Warehouse | Warehouse Manager, checks stock urgency and constraints |
| Supplier  | Supplier Representative, quotes prices & risks          |
| Logistics | Transport Manager, plans deliveries & flags delays      |
| Legal     | Compliance Officer, flags compliance issues             |

---

## ⚡ Notes

* Simulation is **fully local** if using Ollama LLM.
