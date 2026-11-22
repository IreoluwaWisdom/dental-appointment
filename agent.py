import json
import os
from datetime import date

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI, OpenAIError


load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
MAX_AGENT_STEPS = 5
MAX_DAILY_LLM_CALLS = 10
MAX_EXTRACTION_ATTEMPTS = 3
LLM_TIMEOUT_SECONDS = 60.0
LIMIT_REACHED_MESSAGE = (
    "You've reached today's AI request limit. Please try again tomorrow."
)
TIMEOUT_MESSAGE = (
    "The AI service took too long to respond. Please try again."
)
LLM_SERVICE_ERROR_MESSAGE = (
    "An error occurred while contacting the AI service. Please try again."
)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    max_retries=0,
)


class LLMCallLimitError(Exception):
    """Raised when a session has used all of its LLM calls for the day."""


class LLMRequestTimeoutError(Exception):
    """Raised when an LLM request exceeds the configured timeout."""


class LLMServiceError(Exception):
    """Raised when the LLM provider returns a non-timeout API error."""


def reset_daily_llm_count_if_needed(session):
    """Reset the session's LLM counter when the calendar date changes."""
    today = date.today().isoformat()
    if session["llm_call_date"] != today:
        session["llm_call_date"] = today
        session["llm_call_count"] = 0


def call_llm(session, **request_options):
    """Check the daily budget, count the attempt, and call the LLM."""
    reset_daily_llm_count_if_needed(session)

    if session["llm_call_count"] >= MAX_DAILY_LLM_CALLS:
        raise LLMCallLimitError(LIMIT_REACHED_MESSAGE)

    session["llm_call_count"] += 1
    request_options.setdefault("timeout", LLM_TIMEOUT_SECONDS)

    try:
        return client.chat.completions.create(**request_options)
    except APITimeoutError as error:
        raise LLMRequestTimeoutError(TIMEOUT_MESSAGE) from error
    except OpenAIError as error:
        raise LLMServiceError(LLM_SERVICE_ERROR_MESSAGE) from error

# This in-memory schedule is reset whenever the program restarts.
appointments = {
    "2026-12-10": ["9:00 AM", "1:00 PM"],
    "2026-12-16": ["9:00 AM", "1:00 PM"],
    "2026-12-26": ["10:00 AM", "2:00 PM", "4:30 PM"],
}


def check_available_slots(appointment_date):
    """Return a copy of the available slots for a date."""
    return appointments.get(appointment_date, []).copy()


def list_available_appointments(start_date=None, end_date=None):
    """Return non-empty future appointments, optionally within an inclusive range."""
    today = date.today()

    try:
        range_start = date.fromisoformat(start_date) if start_date else today
        range_end = date.fromisoformat(end_date) if end_date else None
    except (TypeError, ValueError):
        return {
            "success": False,
            "message": "Use dates in YYYY-MM-DD format.",
            "appointments": {},
        }

    range_start = max(range_start, today)

    if range_end is not None and range_end < range_start:
        return {
            "success": False,
            "message": "The end date must not be before the start date.",
            "appointments": {},
        }

    available_appointments = {}
    for appointment_date in sorted(appointments):
        parsed_date = date.fromisoformat(appointment_date)
        available_slots = appointments[appointment_date]

        if parsed_date < range_start:
            continue
        if range_end is not None and parsed_date > range_end:
            continue
        if not available_slots:
            continue

        available_appointments[appointment_date] = available_slots.copy()

    if not available_appointments:
        return {
            "success": True,
            "message": "There are no appointments in the requested range.",
            "appointments": {},
        }

    return {
        "success": True,
        "message": "Available appointments found.",
        "appointments": available_appointments,
    }


def validate_appointment_date(date_string):
    """Check that a value is an ISO date and is not in the past."""
    try:
        appointment_date = date.fromisoformat(date_string)
    except (TypeError, ValueError):
        return {"valid": False, "message": "Use a valid date in YYYY-MM-DD format."}

    if appointment_date < date.today():
        return {"valid": False, "message": "Cannot book a past date."}
    return {"valid": True, "message": "Date validated."}


def book_appointment(appointment_date, appointment_time, patient_name):
    """Reserve an available slot after validating the requested date."""
    validation = validate_appointment_date(appointment_date)
    if not validation["valid"]:
        return {"success": False, "message": f"Booking failed: {validation['message']}"}

    available_slots = appointments.get(appointment_date)
    if available_slots is None:
        return {"success": False, "message": "No appointments are offered on this date."}
    if appointment_time not in available_slots:
        return {"success": False, "message": "This time is not available."}

    available_slots.remove(appointment_time)
    return {
        "success": True,
        "message": (
            f"Hello {patient_name}, your appointment is confirmed for "
            f"{appointment_date} at {appointment_time}."
        ),
    }


def create_session():
    """Create the state kept for one conversation."""
    return {
        "messages": [],
        "pending_booking": None,
        "awaiting_modification": False,
        "llm_call_count": 0,
        "llm_call_date": date.today().isoformat(),
    }


def set_pending_booking(session, appointment_date, appointment_time, patient_name):
    """Prepare a valid booking proposal without reserving the slot."""
    validation = validate_appointment_date(appointment_date)
    if not validation["valid"]:
        return {"success": False, "message": validation["message"]}
    if appointment_time not in check_available_slots(appointment_date):
        return {
            "success": False,
            "message": "That slot is not available. Please choose an available date and time.",
        }

    session["pending_booking"] = {
        "date": appointment_date,
        "time": appointment_time,
        "patient_name": patient_name,
    }
    session["awaiting_modification"] = False
    return {
        "success": True,
        "message": (
            f"{patient_name}, please confirm your appointment for "
            f"{appointment_date} at {appointment_time}. Should I proceed?"
        ),
    }


def confirm_booking(pending_booking, user_confirmed):
    """Book a proposal only when the user explicitly confirmed it."""
    if pending_booking is None:
        return {"success": False, "message": "No pending booking exists."}
    if not user_confirmed:
        return {"success": False, "message": "Booking cancelled; the slot is unchanged."}
    return book_appointment(
        pending_booking["date"], pending_booking["time"], pending_booking["patient_name"]
    )


def process_booking_confirmation(session, user_confirmed):
    """Finish or cancel the pending proposal and clear it from the session."""
    pending = session["pending_booking"]
    result = confirm_booking(pending, user_confirmed)
    if pending is not None:
        session["pending_booking"] = None
        session["awaiting_modification"] = False
    return result


def modify_pending_booking(session, new_date=None, new_time=None):
    """Change supplied fields on a proposal after validating the resulting slot."""
    pending = session["pending_booking"]
    if pending is None:
        return {"success": False, "message": "No pending booking exists."}

    updated_date = new_date or pending["date"]
    updated_time = new_time or pending["time"]
    validation = validate_appointment_date(updated_date)
    if not validation["valid"]:
        return {"success": False, "message": validation["message"]}
    if updated_time not in check_available_slots(updated_date):
        return {"success": False, "message": "That date and time combination is not available."}

    pending["date"] = updated_date
    pending["time"] = updated_time
    session["awaiting_modification"] = False
    return {
        "success": True,
        "message": (
            f"{pending['patient_name']}, your proposed appointment is now "
            f"{updated_date} at {updated_time}. Would you like to confirm it?"
        ),
    }


def extract_booking_changes(user_message, session):
    """Use an LLM to extract normalized booking changes from natural language."""
    extraction_prompt = f"""
You extract requested changes to a dental appointment.
Today's date is {date.today().isoformat()}.

Return one JSON object with exactly these keys:
- "date": the requested date in YYYY-MM-DD format, or null if no date was given
- "time": the requested time in H:MM AM/PM format, or null if no time was given

Resolve relative dates such as "tomorrow" using today's date.
Do not invent a date or time that the user did not request.
Do not include explanations or additional keys.
"""

    for _ in range(MAX_EXTRACTION_ATTEMPTS):
        try:
            response = call_llm(
                session,
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            extracted_changes = json.loads(response.choices[0].message.content)
        except LLMCallLimitError:
            raise
        except LLMRequestTimeoutError:
            raise
        except LLMServiceError:
            raise
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue

        if not isinstance(extracted_changes, dict):
            continue

        new_date = extracted_changes.get("date")
        new_time = extracted_changes.get("time")

        if new_date is not None and not isinstance(new_date, str):
            new_date = None
        if new_time is not None and not isinstance(new_time, str):
            new_time = None

        if new_date is not None or new_time is not None:
            return new_date, new_time

    return None, None


def classify_booking_response(user_message, session):
    """Classify a reply to a proposed booking into one allowed intent."""
    classification_prompt = """
You classify a patient's response to a proposed appointment.
Return exactly one intent: CONFIRM, CANCEL, MODIFY, or UNCLEAR.
CONFIRM means explicit, unconditional agreement to book.
CANCEL means an explicit decline or cancellation.
MODIFY means a request to change the proposed booking.
UNCLEAR means ambiguous, unrelated, contradictory, or conditional.
Do not execute a booking. Return only the intent word.
"""
    response = call_llm(
        session,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    classification = response.choices[0].message.content.strip().upper()
    allowed_intents = {"CONFIRM", "CANCEL", "MODIFY", "UNCLEAR"}
    return classification if classification in allowed_intents else "UNCLEAR"


def handle_booking_response(user_message, session):
    """Route a user's reply while a booking proposal is pending."""
    if session["pending_booking"] is None:
        return {"success": False, "message": "There is no booking awaiting confirmation."}

    if session["awaiting_modification"]:
        new_date, new_time = extract_booking_changes(user_message, session)
        if new_date is None and new_time is None:
            return {
                "success": False,
                "message": "Please provide a date as YYYY-MM-DD, a time such as 2:00 PM, or both.",
            }
        return modify_pending_booking(session, new_date, new_time)

    classification = classify_booking_response(user_message, session)
    if classification == "CONFIRM":
        return process_booking_confirmation(session, True)
    if classification == "CANCEL":
        return process_booking_confirmation(session, False)
    if classification == "MODIFY":
        session["awaiting_modification"] = True
        return {
            "success": True,
            "message": "What would you like to change? Provide a new date, time, or both.",
        }
    return {
        "success": False,
        "message": "Please clearly confirm, cancel, or ask to change the proposed booking.",
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "check_available_slots",
            "description": "Check available appointment times for a particular date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."}
                },
                "required": ["date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_appointments",
            "description": (
                "List every future appointment date with open slots, or list open "
                "appointments within an inclusive date range. Use this for general "
                "availability, 'this week', or 'the next seven days'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": ["string", "null"],
                        "description": "Optional inclusive start date in YYYY-MM-DD format.",
                    },
                    "end_date": {
                        "type": ["string", "null"],
                        "description": "Optional inclusive end date in YYYY-MM-DD format.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_pending_booking",
            "description": "Validate and prepare an appointment for confirmation without reserving it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "time": {"type": "string", "description": "Time, for example 2:00 PM."},
                    "patient_name": {"type": "string", "description": "Patient's name."},
                },
                "required": ["date", "time", "patient_name"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool_call(tool_call, session):
    """Run one allow-listed tool call requested by the model."""
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return {"success": False, "message": "The tool arguments were not valid JSON."}

    function_name = tool_call.function.name
    try:
        if function_name == "check_available_slots":
            return {
                "success": True,
                "date": arguments["date"],
                "available_slots": check_available_slots(arguments["date"]),
            }
        if function_name == "list_available_appointments":
            return list_available_appointments(
                arguments.get("start_date"),
                arguments.get("end_date"),
            )
        if function_name == "set_pending_booking":
            return set_pending_booking(
                session, arguments["date"], arguments["time"], arguments["patient_name"]
            )
    except KeyError as error:
        return {"success": False, "message": f"Missing tool argument: {error.args[0]}"}
    return {"success": False, "message": f"Unknown tool: {function_name}"}


def run_agent(user_message, session):
    """Process one user turn, allowing a bounded number of model/tool steps."""
    session["messages"].append({"role": "user", "content": user_message})

    if session["pending_booking"] is not None:
        try:
            result = handle_booking_response(user_message, session)
        except LLMCallLimitError:
            result = {"success": False, "message": LIMIT_REACHED_MESSAGE}
        except LLMRequestTimeoutError:
            result = {"success": False, "message": TIMEOUT_MESSAGE}
        except LLMServiceError:
            result = {"success": False, "message": LLM_SERVICE_ERROR_MESSAGE}
        session["messages"].append({"role": "assistant", "content": result["message"]})
        return result["message"]

    if len(session["messages"]) == 1:
        session["messages"].insert(
            0,
            {
                "role": "system",
                "content": (
                    f"You are a dental appointment assistant. Today is "
                    f"{date.today().isoformat()}. Use tools whenever appointment information "
                    "is needed; never invent slots. Use check_available_slots for one exact "
                    "date. Use list_available_appointments for general availability or a date "
                    "range. Interpret 'this week' as Monday through Sunday of the current week. "
                    "Interpret 'the next seven days' as today through six days from today. Use "
                    "set_pending_booking when all booking details are supplied. A pending booking "
                    "is only a proposal. Never claim a booking is complete until Python confirms "
                    "it after explicit agreement."
                ),
            },
        )

    for _ in range(MAX_AGENT_STEPS):
        try:
            response = call_llm(
                session,
                model=MODEL_NAME,
                messages=session["messages"],
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
        except LLMCallLimitError:
            session["messages"].append(
                {"role": "assistant", "content": LIMIT_REACHED_MESSAGE}
            )
            return LIMIT_REACHED_MESSAGE
        except LLMRequestTimeoutError:
            session["messages"].append(
                {"role": "assistant", "content": TIMEOUT_MESSAGE}
            )
            return TIMEOUT_MESSAGE
        except LLMServiceError:
            session["messages"].append(
                {"role": "assistant", "content": LLM_SERVICE_ERROR_MESSAGE}
            )
            return LLM_SERVICE_ERROR_MESSAGE
        message = response.choices[0].message
        session["messages"].append(message)
        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            tool_result = execute_tool_call(tool_call, session)
            session["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                }
            )

    failure_message = "I could not complete that request within the allowed steps."
    session["messages"].append({"role": "assistant", "content": failure_message})
    return failure_message


if __name__ == "__main__":
    conversation = create_session()
    print("Dental appointment assistant. Type 'quit' to stop.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input:
            print(f"Assistant: {run_agent(user_input, conversation)}")
