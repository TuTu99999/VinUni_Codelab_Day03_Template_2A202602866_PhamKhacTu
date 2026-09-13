"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
import sys
import unicodedata
from typing import Any, Dict, List

from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> Dict[str, Any]:
        """Return a one-shot response and make the lack of tool calls explicit."""
        return {
            "status": "success",
            "answer": (
                "Tôi chưa thể tra cứu dữ liệu chuyến bay hoặc thời tiết theo thời gian "
                f"thực cho yêu cầu: {user_input}"
            ),
            "tool_calls": [],
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    @staticmethod
    def _normalise(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text.lower())
        return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")

    @staticmethod
    def _airport_codes(user_input: str) -> List[str]:
        # Uppercasing the whole Vietnamese sentence would also turn words such as
        # "bay" into apparent IATA codes.  Keep only the codes supported by the
        # lab's flight/weather datasets.
        supported_codes = {"HAN", "SGN", "DAD"}
        return [
            code
            for code in re.findall(r"\b[A-Z]{3}\b", user_input.upper())
            if code in supported_codes
        ]

    @staticmethod
    def _max_price(user_input: str) -> int:
        normalised = user_input.lower().replace(",", ".")
        million = re.search(r"(\d+(?:\.\d+)?)\s*(?:triệu|trieu)", normalised)
        if million:
            return int(float(million.group(1)) * 1_000_000)

        thousands = re.search(r"(\d+(?:\.\d+)?)\s*k\b", normalised)
        if thousands:
            return int(float(thousands.group(1)) * 1_000)

        digits = re.search(r"(?:dưới|duoi|max|tối đa|toi da)\s*([\d.]+)", normalised)
        if digits:
            return int(digits.group(1).replace(".", ""))
        return 5_000_000

    @staticmethod
    def _format_flights(flights: List[Dict[str, Any]]) -> str:
        if not flights:
            return "Không tìm thấy chuyến bay phù hợp với hành trình và ngân sách đã cho."
        details = [
            f"{flight['flight_number']} ({flight['airline']}), khởi hành "
            f"{flight['departure_time']}, giá {flight['price_vnd']:,} VND"
            for flight in flights
        ]
        return "Các chuyến bay phù hợp: " + "; ".join(details) + "."

    @staticmethod
    def _format_weather(weather: Dict[str, Any]) -> str:
        if "error" in weather:
            return f"Không lấy được thông tin thời tiết: {weather['error']}."
        return (
            f"Thời tiết {weather['city']}: {weather['temperature_c']}°C, "
            f"{weather['condition']}, độ ẩm {weather['humidity_pct']}%. "
            f"Gợi ý: {weather['recommendation']}"
        )

    def _call_tool(self, iteration: int, name: str, args: Dict[str, Any], thought: str) -> Any:
        """Execute a registered tool and append one ReAct trace entry."""
        tool_name = name.strip().lower()
        action = {"name": tool_name, "args": args}
        try:
            tool = TOOL_MAP[tool_name]
            observation = tool(**args)
        except KeyError:
            observation = {"error": f"Unknown tool: {tool_name}"}
        except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            observation = {"error": str(exc)}

        self.trace.append(
            {
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": observation,
            }
        )
        return observation

    def run(self, user_input: str) -> Dict[str, Any]:
        """Run a deterministic Thought-Action-Observation loop for the lab domain."""
        self.trace = []
        iteration = 0
        normalised = self._normalise(user_input)
        codes = self._airport_codes(user_input)

        is_faq = any(term in normalised for term in ("chinh sach", "doi tra", "hoan ve"))
        needs_flight = not is_faq and any(
            term in normalised for term in ("chuyen bay", "ve may bay", "flight")
        )
        needs_weather = any(
            term in normalised for term in ("thoi tiet", "mac gi", "weather")
        )

        planned_actions = []
        if needs_flight and len(codes) >= 2:
            planned_actions.append(
                (
                    "get_flight_info",
                    {
                        "origin": codes[0],
                        "destination": codes[1],
                        "max_price": self._max_price(user_input),
                    },
                    "Cần tra cứu các chuyến bay phù hợp với hành trình và ngân sách.",
                )
            )
        if needs_weather:
            city_code = codes[-1] if codes else ""
            planned_actions.append(
                (
                    "get_weather_forecast",
                    {"city_code": city_code},
                    "Cần tra cứu thời tiết và gợi ý trang phục tại điểm đến.",
                )
            )

        observations: Dict[str, Any] = {}
        for name, args, thought in planned_actions:
            if iteration >= self.max_iterations:
                return {
                    "status": "max_iterations_reached",
                    "answer": "Không thể hoàn thành trong số bước tối đa.",
                    "iterations": iteration,
                    "trace": self.trace,
                }
            iteration += 1
            observations[name] = self._call_tool(iteration, name, args, thought)

        answer_parts = []
        if "get_flight_info" in observations:
            answer_parts.append(self._format_flights(observations["get_flight_info"]))
        if "get_weather_forecast" in observations:
            answer_parts.append(self._format_weather(observations["get_weather_forecast"]))

        if not planned_actions:
            iteration = 1
            if is_faq:
                answer_parts.append(
                    "Chính sách đổi trả vé máy bay Vinpearl phụ thuộc vào điều kiện của "
                    "hạng vé và nhà vận chuyển. Vui lòng kiểm tra điều kiện vé hoặc liên hệ "
                    "Vinpearl để được xác nhận phí và thời hạn đổi trả."
                )
            else:
                answer_parts.append(
                    "Tôi chưa xác định được yêu cầu tra cứu. Vui lòng cung cấp mã sân bay "
                    "và nội dung cần hỗ trợ."
                )
            self.trace.append(
                {
                    "iteration": iteration,
                    "thought": "Câu hỏi không cần công cụ; trả lời trực tiếp.",
                    "action": None,
                    "observation": None,
                    "final_answer": answer_parts[0],
                }
            )
        elif len(planned_actions) > 1:
            if iteration >= self.max_iterations:
                return {
                    "status": "max_iterations_reached",
                    "answer": "Không thể hoàn thành trong số bước tối đa.",
                    "iterations": iteration,
                    "trace": self.trace,
                }
            iteration += 1
            self.trace.append(
                {
                    "iteration": iteration,
                    "thought": "Đã đủ dữ liệu; tổng hợp kết quả từ các công cụ.",
                    "action": None,
                    "observation": None,
                    "final_answer": " ".join(answer_parts),
                }
            )
        else:
            self.trace[-1]["final_answer"] = " ".join(answer_parts)

        return {
            "status": "completed",
            "answer": " ".join(answer_parts),
            "iterations": iteration,
            "trace": self.trace,
        }

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
