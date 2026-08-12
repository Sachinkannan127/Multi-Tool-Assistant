"""
Code Generator Tool — generates clean, production-ready code for any programming language.
Uses the current LLM (Groq / Gemini) via a structured prompt.
"""

from langchain_core.tools import tool


SUPPORTED_LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "Go",
    "Rust", "Swift", "Kotlin", "Ruby", "PHP", "Scala", "R", "Dart",
    "Bash", "PowerShell", "SQL", "PostgreSQL", "MySQL", "MongoDB",
    "HTML", "CSS", "SCSS", "React", "Vue", "Svelte",
    "YAML", "JSON", "Dockerfile", "Terraform", "Ansible",
]


def build_code_prompt(language: str, description: str) -> str:
    """Build a precise, structured prompt for code generation."""
    return f"""Generate clean, production-ready {language} code for the following requirement:

REQUIREMENT:
{description}

INSTRUCTIONS:
- Write ONLY the code — no markdown fences, no explanations outside comments
- Add concise inline comments explaining key logic
- Follow {language} best practices and idiomatic style
- Include a brief docstring / header comment describing what the code does
- Make the code complete and runnable (include imports, function signatures, etc.)
- Do NOT wrap the code in triple backticks

CODE:"""


@tool
def code_generator_tool(language_and_description: str) -> str:
    """Generate clean, production-ready code in any programming language.

    Use this tool whenever the user asks you to:
    - Write, create, generate, or implement code
    - Build a function, class, script, or module
    - Convert logic to a specific programming language
    - Provide an example implementation

    Input format: "<Language>: <description of what to code>"
    Example: "Python: bubble sort algorithm with unit tests"
    Example: "TypeScript: fetch wrapper with retry logic and timeout"
    Example: "SQL: find top 5 customers by revenue in the last 30 days"
    """
    from config import settings
    from langchain_core.messages import HumanMessage

    try:
        # Parse language and description from input
        if ":" in language_and_description:
            language, description = language_and_description.split(":", 1)
            language = language.strip()
            description = description.strip()
        else:
            language = "Python"
            description = language_and_description.strip()

        if not description:
            return "Error: Please provide a description of what code to generate."

        # Build the LLM
        if settings.llm_provider in ["google", "gemini"]:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.google_api_key,
                temperature=0.2,
            )
        else:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=settings.groq_model,
                groq_api_key=settings.groq_api_key,
                temperature=0.2,
            )

        prompt = build_code_prompt(language, description)
        response = llm.invoke([HumanMessage(content=prompt)])
        code = response.content.strip()

        # Strip any accidental markdown fences the LLM added anyway
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first line (```language) and last line (```)
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        return f"**{language} Code:**\n\n```{language.lower()}\n{code}\n```"

    except Exception as e:
        return f"Code generation failed: {str(e)}"
