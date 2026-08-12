"""
Secure Calculator Tool using Python AST parsing.
No eval() or exec() — only safe mathematical operations are allowed.
"""

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool


# Allowed operators mapping
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math functions
SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pow": pow,
    "min": min,
    "max": max,
    "sum": sum,
}

# Mathematical constants
SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Unit conversion factors (to base SI units)
CONVERSIONS = {
    # Length (to meters)
    "km": 1000, "m": 1, "cm": 0.01, "mm": 0.001, "mi": 1609.344,
    "yd": 0.9144, "ft": 0.3048, "in": 0.0254, "nm": 1852,
    # Weight (to grams)
    "kg": 1000, "g": 1, "mg": 0.001, "lb": 453.592, "oz": 28.3495, "ton": 907185,
    # Temperature (special handling)
    # Volume (to liters)
    "l": 1, "ml": 0.001, "gal": 3.78541, "qt": 0.946353, "cup": 0.236588,
    "floz": 0.0295735, "tbsp": 0.0147868, "tsp": 0.00492892,
    # Time (to seconds)
    "s": 1, "min": 60, "hr": 3600, "day": 86400, "week": 604800,
    "month": 2592000, "year": 31536000,
    # Data (to bytes)
    "b": 1, "kb": 1024, "mb": 1048576, "gb": 1073741824, "tb": 1099511627776,
}


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node safely."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    elif isinstance(node, ast.Num):  # Python 3.7 compat
        return node.n

    elif isinstance(node, ast.Name):
        name = node.id.lower()
        if name in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[name]
        raise ValueError(f"Unknown variable or constant: '{node.id}'. Available: {list(SAFE_CONSTANTS.keys())}")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        # Safety: prevent excessively large powers
        if op_type == ast.Pow and isinstance(right, (int, float)) and right > 1000:
            raise ValueError("Exponent too large (max 1000)")
        return SAFE_OPERATORS[op_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        func_name = node.func.id.lower()
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' not allowed. Available: {list(SAFE_FUNCTIONS.keys())}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return SAFE_FUNCTIONS[func_name](*args)

    elif isinstance(node, ast.IfExp):
        test = _safe_eval_node(node.test)
        if test:
            return _safe_eval_node(node.body)
        return _safe_eval_node(node.orelse)

    elif isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = _safe_eval_node(comparator)
            if isinstance(op, ast.Lt):
                if not (left < right):
                    return False
            elif isinstance(op, ast.LtE):
                if not (left <= right):
                    return False
            elif isinstance(op, ast.Gt):
                if not (left > right):
                    return False
            elif isinstance(op, ast.GtE):
                if not (left >= right):
                    return False
            elif isinstance(op, ast.Eq):
                if not (left == right):
                    return False
            elif isinstance(op, ast.NotEq):
                if not (left != right):
                    return False
            else:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            left = right
        return True

    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression using AST parsing.

    Supports:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Math functions: sqrt, sin, cos, tan, log, log10, exp, abs, round, ceil, floor, factorial
    - Constants: pi, e, tau
    - Comparisons: <, >, <=, >=, ==, !=
    """
    expression = expression.strip()
    if not expression:
        return {"error": "Empty expression"}

    # Sanitize: remove any potentially dangerous characters
    sanitized = expression.replace(";", "").replace("__", "").replace("import", "").replace("exec", "").replace("eval", "")

    try:
        tree = ast.parse(sanitized, mode="eval")
        result = _safe_eval_node(tree)
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__,
        }
    except ZeroDivisionError:
        return {"error": "Division by zero", "expression": expression}
    except ValueError as e:
        return {"error": str(e), "expression": expression}
    except Exception as e:
        return {"error": f"Calculation error: {str(e)}", "expression": expression}


def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert between units of measurement."""
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    # Special handling for temperature
    temp_units = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}
    if from_unit in temp_units or to_unit in temp_units:
        return _convert_temperature(value, from_unit, to_unit)

    if from_unit not in CONVERSIONS:
        return {"error": f"Unknown unit: '{from_unit}'. Available: {list(CONVERSIONS.keys())}"}
    if to_unit not in CONVERSIONS:
        return {"error": f"Unknown unit: '{to_unit}'. Available: {list(CONVERSIONS.keys())}"}

    # Convert to base unit, then to target
    base_value = value * CONVERSIONS[from_unit]
    result = base_value / CONVERSIONS[to_unit]

    return {
        "from": f"{value} {from_unit}",
        "to": f"{result:.6g} {to_unit}",
        "result": result,
    }


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert between temperature units."""
    from_u = from_unit.lower().replace("celsius", "c").replace("fahrenheit", "f").replace("kelvin", "k")
    to_u = to_unit.lower().replace("celsius", "c").replace("fahrenheit", "f").replace("kelvin", "k")

    # Convert to Celsius first
    if from_u == "c":
        celsius = value
    elif from_u == "f":
        celsius = (value - 32) * 5 / 9
    elif from_u == "k":
        celsius = value - 273.15
    else:
        return {"error": f"Unknown temperature unit: {from_unit}"}

    # Convert from Celsius to target
    if to_u == "c":
        result = celsius
    elif to_u == "f":
        result = celsius * 9 / 5 + 32
    elif to_u == "k":
        result = celsius + 273.15
    else:
        return {"error": f"Unknown temperature unit: {to_unit}"}

    return {
        "from": f"{value}°{from_u.upper()}",
        "to": f"{result:.2f}°{to_u.upper()}",
        "result": result,
    }


def financial_calc(principal: float, rate: float, time_years: float, compounds_per_year: int = 1) -> dict:
    """Calculate compound interest."""
    amount = principal * (1 + rate / 100 / compounds_per_year) ** (compounds_per_year * time_years)
    interest = amount - principal
    return {
        "principal": principal,
        "rate_percent": rate,
        "time_years": time_years,
        "compounds_per_year": compounds_per_year,
        "final_amount": round(amount, 2),
        "total_interest": round(interest, 2),
    }


@tool
def calculator_tool(query: str) -> str:
    """
    A secure calculator tool for mathematical expressions, unit conversions, and financial calculations.

    Input should be one of:
    - A math expression: "2 + 2", "sqrt(144)", "sin(pi/4)", "2**10"
    - A unit conversion: "convert 100 km to miles" or "convert 72 f to c"
    - A financial calculation: "compound interest 10000 at 5% for 3 years"

    Supports: +, -, *, /, //, %, **, sqrt, sin, cos, tan, log, log10, exp, abs, round, ceil, floor, factorial
    Constants: pi, e, tau
    Unit categories: length (km, m, cm, mm, mi, yd, ft, in), weight (kg, g, mg, lb, oz),
                     volume (l, ml, gal, qt, cup), time (s, min, hr, day, week, year),
                     data (b, kb, mb, gb, tb), temperature (c, f, k)
    """
    query_lower = query.strip().lower()

    # Check for unit conversion
    if "convert" in query_lower or " to " in query_lower:
        try:
            parts = query_lower.replace("convert", "").strip().split(" to ")
            if len(parts) == 2:
                value_and_from = parts[0].strip().split()
                to_unit = parts[1].strip()
                if len(value_and_from) >= 2:
                    value = float(value_and_from[0])
                    from_unit = value_and_from[1]
                    result = convert_units(value, from_unit, to_unit)
                    if "error" in result:
                        return f"Conversion error: {result['error']}"
                    return f"{result['from']} = {result['to']}"
        except (ValueError, IndexError):
            pass

    # Check for financial calculation
    if "compound interest" in query_lower or "interest" in query_lower:
        try:
            import re
            numbers = re.findall(r'[\d.]+', query)
            if len(numbers) >= 3:
                principal = float(numbers[0])
                rate = float(numbers[1])
                time = float(numbers[2])
                compounds = int(numbers[3]) if len(numbers) > 3 else 1
                result = financial_calc(principal, rate, time, compounds)
                return (
                    f"Compound Interest Calculation:\n"
                    f"Principal: ${result['principal']:,.2f}\n"
                    f"Rate: {result['rate_percent']}%\n"
                    f"Time: {result['time_years']} years\n"
                    f"Final Amount: ${result['final_amount']:,.2f}\n"
                    f"Total Interest Earned: ${result['total_interest']:,.2f}"
                )
        except (ValueError, IndexError):
            pass

    # Standard math expression
    result = safe_calculate(query)
    if "error" in result:
        return f"Error: {result['error']}"

    res = result["result"]
    if isinstance(res, float):
        # Format nicely
        if res == int(res) and abs(res) < 1e15:
            return f"{result['expression']} = {int(res)}"
        return f"{result['expression']} = {res:.10g}"
    return f"{result['expression']} = {res}"
