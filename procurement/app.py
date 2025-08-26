"""
Zimbabwean Procurement Internship Simulation
--------------------------------------------------------
A web-based UI using Streamlit to simulate a multi-agent procurement scenario.
Each day, the intern (user) makes a decision, and AI agents evaluate it.
"""

import streamlit as st
from dataclasses import dataclass
from typing import List, Dict
from crewai import Agent, Task, Crew
from langchain_community.chat_models import ChatOllama

# -------------------------
# Scenario model
# -------------------------

@dataclass
class SupplierOption:
    name: str
    prices: Dict[str, float]
    delivery: str
    reliability_note: str = ""

@dataclass
class DayScenario:
    title: str
    warehouse_request: Dict[str, int]
    finance_budget: float
    supplier_options: List[SupplierOption]
    special_event: str = ""

# -------------------------
# Sample Scenarios
# -------------------------

SCENARIOS: List[DayScenario] = [
    DayScenario(
        title="Day 1 — Stationery Procurement",
        warehouse_request={"A4_paper_box": 20, "pen_box": 10},
        finance_budget=500.0,
        supplier_options=[
            SupplierOption(
                name="Harare OfficeSupplies",
                prices={"A4_paper_box": 18.0, "pen_box": 8.0},
                delivery="2 days",
                reliability_note="Often on-time, occasional stockouts",
            ),
            SupplierOption(
                name="SA Importers",
                prices={"A4_paper_box": 15.0, "pen_box": 10.0},
                delivery="7 days",
                reliability_note="Slower, cross-border delays possible",
            ),
        ],
    ),
    DayScenario(
        title="Day 2 — Fresh Produce Replenishment",
        warehouse_request={"tomatoes_crate": 15, "onions_bag": 8},
        finance_budget=700.0,
        supplier_options=[
            SupplierOption(
                name="Chegutu Farmer Co-op",
                prices={"tomatoes_crate": 18.0, "onions_bag": 12.0},
                delivery="Same day",
                reliability_note="Limited volume, quality varies",
            ),
            SupplierOption(
                name="Mbare Wholesale",
                prices={"tomatoes_crate": 22.0, "onions_bag": 14.0},
                delivery="Immediate pickup",
                reliability_note="Reliable availability, higher price",
            ),
        ],
        special_event="Sudden demand spike from a large retail client",
    ),
    DayScenario(
        title="Day 3 — Compliance & Logistics",
        warehouse_request={"toner_cartridge": 6},
        finance_budget=400.0,
        supplier_options=[
            SupplierOption(
                name="QuickPrint Supplies",
                prices={"toner_cartridge": 70.0},
                delivery="3 days",
                reliability_note="Claims tax clearance; verify validity",
            ),
            SupplierOption(
                name="BudgetPrint",
                prices={"toner_cartridge": 60.0},
                delivery="5 days",
                reliability_note="No recent ZIMRA tax clearance on file",
            ),
        ],
        special_event="Primary delivery truck broke down en route",
    ),
]

# -------------------------
# LLM Setup
# -------------------------

LLM = ChatOllama(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434"
)

# -------------------------
# Agent Builders
# -------------------------

def build_mentor_agent():
    return Agent(
        name="Mentor",
        role="Senior Procurement Officer",
        goal="Evaluate intern decisions and give feedback with a score out of 10.",
        backstory="You coach interns with concise, actionable feedback.",
        llm=LLM,
        verbose=True,
    )

def build_finance_agent():
    return Agent(
        name="Finance",
        role="Finance Officer",
        goal="Assess budget fit and suggest cost-saving measures. keep it under 100 words",
        backstory="Strict but fair budget manager.",
        llm=LLM,
        verbose=True,
    )

def build_warehouse_agent():
    return Agent(
        name="Warehouse",
        role="Warehouse Manager",
        goal="Report stock urgency and receiving constraints. keep it under 100 words",
        backstory="Tracks goods pragmatically.",
        llm=LLM,
        verbose=True,
    )

def build_supplier_agent():
    return Agent(
        name="Supplier",
        role="Supplier Representative",
        goal="Quote prices and warn about risks.keep it under 100 words",
        backstory="Commercially motivated supplier rep.",
        llm=LLM,
        verbose=True,
    )

def build_logistics_agent():
    return Agent(
        name="Logistics",
        role="Transport & Delivery Manager",
        goal="Plan delivery and flag delays.keep it under 100 words",
        backstory="Manages fuel and road constraints.",
        llm=LLM,
        verbose=True,
    )

def build_legal_agent():
    return Agent(
        name="Legal",
        role="Compliance Officer",
        goal="Check compliance and flag red flags. keep it under 100 words",
        backstory="Enforces local procurement laws.",
        llm=LLM,
        verbose=True,
    )

# -------------------------
# Prompt Helpers
# -------------------------

def scenario_brief(day: DayScenario) -> str:
    lines = [
        f"Scenario: {day.title}",
        "\nWarehouse request:",
        *(f"- {k}: {v}" for k, v in day.warehouse_request.items()),
        f"\nFinance budget: USD {day.finance_budget}",
        "\nSupplier options:",
    ]
    for s in day.supplier_options:
        price_str = ", ".join(f"{k} ${v:.2f}" for k, v in s.prices.items())
        lines.append(f"- {s.name}: {price_str} | Delivery {s.delivery} | {s.reliability_note}")
    if day.special_event:
        lines.append(f"\nSpecial event: {day.special_event}")
    return "\n".join(lines)

def mentor_eval_prompt(user_decision: str, day: DayScenario) -> str:
    return (
        "You are the Mentor evaluating a procurement intern's decision.\n"
        f"Intern decision:\n{user_decision}\n\n"
        f"Context:\n{scenario_brief(day)}\n\n"
        "Provide: (1) 2–3 bullet strengths, (2) 2–3 bullet improvements, "
        "(3) risk/compliance notes, (4) a score /10. Keep it under 80 words."
    )

# -------------------------
# Custom Crew with Real-time Output
# -------------------------

class StreamlitCrew:
    def __init__(self, agents, tasks, process="sequential"):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.results = {}
        
    def kickoff(self):
        # Create placeholders for each agent
        placeholders = {}
        cols = st.columns(2)
        
        # Map agent roles to columns
        col_mapping = {
            "Finance Officer": cols[0],
            "Warehouse Manager": cols[0],
            "Legal Officer": cols[0],
            "Supplier Representative": cols[1],
            "Transport & Delivery Manager": cols[1],
            "Senior Procurement Officer": cols[1]
        }
        
        # Create placeholders for each agent
        for agent in self.agents:
            role = agent.role
            if role in col_mapping:
                with col_mapping[role]:
                    st.markdown(f"### {self._get_agent_icon(role)} {role}")
                    placeholders[agent.role] = st.empty()
        
        # Run tasks sequentially and update placeholders
        final_results = []
        for i, task in enumerate(self.tasks):
            agent_role = task.agent.role
            placeholder = placeholders.get(agent_role)
            
            if placeholder:
                with placeholder.container():
                    with st.spinner("Thinking..."):
                        # Execute the task
                        result = task.execute_sync()
                        self.results[agent_role] = result
                        
                        # Get the output content (fixed attribute access)
                        output_content = self._get_output_content(result)
                        
                        # Display result with appropriate styling
                        if "Mentor" in agent_role:
                            st.success(output_content)
                        elif "Legal" in agent_role or "Finance" in agent_role:
                            st.warning(output_content)
                        else:
                            st.info(output_content)
                        
                        final_results.append(output_content)
            
        return "\n".join(final_results)
    
    def _get_output_content(self, result):
        """Extract content from task output safely"""
        if hasattr(result, 'raw_output'):
            return result.raw_output
        elif hasattr(result, 'result'):
            return result.result
        elif hasattr(result, 'content'):
            return result.content
        else:
            return str(result)
    
    def _get_agent_icon(self, role):
        icons = {
            "Finance Officer": "💰",
            "Warehouse Manager": "📦",
            "Supplier Representative": "🛒",
            "Transport & Delivery Manager": "🚚",
            "Legal Officer": "⚖️",
            "Senior Procurement Officer": "🎓"
        }
        return icons.get(role, "🤖")

# -------------------------
# Run Simulation
# -------------------------

def run_simulation(day: DayScenario, user_decision: str):
    mentor = build_mentor_agent()
    finance = build_finance_agent()
    warehouse = build_warehouse_agent()
    supplier = build_supplier_agent()
    logistics = build_logistics_agent()
    legal = build_legal_agent()

    finance_task = Task(
        description=f"Evaluate the intern's proposed order for budget fit and cash flow impact.\nIntern decision: {user_decision}",
        agent=finance,
        expected_output="Approval status with concise rationale.",
    )

    warehouse_task = Task(
        description=f"Assess how the decision meets operational needs.\nIntern decision: {user_decision}",
        agent=warehouse,
        expected_output="Operational assessment bullets.",
    )

    supplier_task = Task(
        description=f"Respond as supplier(s) with quote confirmation.\nIntern decision: {user_decision}",
        agent=supplier,
        expected_output="Supplier response with terms & caveats.",
    )

    logistics_task = Task(
        description=f"Propose delivery plan and flag delays.\nIntern decision: {user_decision}\nSpecial event: {day.special_event}",
        agent=logistics,
        expected_output="Delivery plan with ETA and risks.",
    )

    legal_task = Task(
        description=f"Check for compliance risks.\nIntern decision: {user_decision}",
        agent=legal,
        expected_output="Compliance notes and recommendations.",
    )

    mentor_task = Task(
        description=mentor_eval_prompt(user_decision, day),
        agent=mentor,
        expected_output="Mentor feedback with score /10."
    )

    crew = StreamlitCrew(
        agents=[finance, warehouse, supplier, logistics, legal, mentor],
        tasks=[finance_task, warehouse_task, supplier_task, logistics_task, legal_task, mentor_task],
        process="sequential"
    )

    return crew.kickoff()

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Zimbabwean Procurement Internship", layout="wide")
st.title("🎓 Zimbabwean Procurement Internship Simulation")
st.markdown("You're a procurement intern at Company X, this is your first workday. Your superviser has assigned work for you to do")

# Sidebar
st.sidebar.header("Choose a Scenario")
day_titles = [d.title for d in SCENARIOS]
selected_day_title = st.sidebar.selectbox("Select Day", day_titles)
selected_day = next(d for d in SCENARIOS if d.title == selected_day_title)

# Display Scenario
st.subheader(selected_day.title)
st.text_area("Scenario Details", value=scenario_brief(selected_day), height=300, disabled=True)

# User Input
user_input = st.text_area("Your Decision (Supplier, quantities, justification)", height=150)

if st.button("Submit Decision"):
    if not user_input.strip():
        st.warning("Please enter a decision.")
    else:
        with st.spinner("Agents are evaluating your decision..."):
            result = run_simulation(selected_day, user_input)

        # Display final result summary
        st.subheader("📋 Summary")
        st.markdown(result)