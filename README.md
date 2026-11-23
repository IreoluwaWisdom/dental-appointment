# Dental Appointment Agent

A conversational dental appointment assistant built with Python, an OpenAI-compatible Groq endpoint, tool calling, and Streamlit.

The agent can show available appointment dates, check a specific date, prepare a booking proposal, and then confirm, cancel, or modify that proposal. Booking is controlled by Python: the language model can request actions, but it cannot reserve a slot by itself.

## Why I built this

This project is my Rung 3 AI engineering project. Its purpose is to practise designing reliable control flow around a non-deterministic language model.

The central safety rule is that a proposed booking is not a completed booking. A slot is removed only after the patient explicitly confirms it and Python validates it again.

## Features

- Displays all future appointment dates and times without calling the LLM
- Checks availability for a specific date
- Lists availability within a date range
- Understands natural-language booking requests
- Creates a pending proposal before booking
- Requires explicit confirmation
- Supports cancellation
- Supports changing the date, time, or both
- Preserves conversation state in Streamlit
- Shows the full visible conversation
- Displays a loader while waiting for the LLM
- Limits the agent loop to five steps
- Limits each session to ten LLM calls per calendar day
- Gives date-and-time extraction up to three attempts
- Stops LLM requests after 60 seconds
- Returns controlled messages for timeouts, provider failures, and usage limits

## Architecture

```text
Streamlit interface (app.py)
        |
        | user message
        v
run_agent() (agent.py)
        |
        | model may request an approved tool
        v
execute_tool_call()
        |
        +--> check_available_slots()
        +--> list_available_appointments()
        +--> set_pending_booking()
        |
        v
Pending proposal
        |
        +--> confirm --> book_appointment()
        +--> cancel  --> leave slot unchanged
        +--> modify  --> extract and validate changes
```

Every LLM request passes through `call_llm()`. This function enforces the daily call limit and timeout before contacting the provider.

## LLM and Python responsibilities

The project deliberately separates probabilistic interpretation from deterministic business rules.

The LLM is responsible for:

- Understanding natural-language requests
- Deciding when an approved tool is needed
- Classifying a reply as `CONFIRM`, `CANCEL`, `MODIFY`, or `UNCLEAR`
- Extracting a requested date and time from natural language

Python is responsible for:

- Storing the available appointment schedule
- Validating date formats and rejecting past dates
- Checking whether a slot actually exists
- Creating and updating pending proposals
- Removing a slot only after explicit confirmation
- Enforcing step, call, retry, and timeout limits

## Available tools

### `check_available_slots`

Returns the open times for one exact date.

### `list_available_appointments`

Returns every non-empty future appointment date, optionally filtered by an inclusive start and end date. Results are sorted chronologically.

### `set_pending_booking`

Validates the requested date and time and stores a proposal in the current session. It does not reserve the slot.

## Project structure

```text
dental-appointment-agent/
|-- agent.py          # Agent, tools, booking logic, and command-line interface
|-- app.py            # Streamlit chat interface
|-- test_agent.py     # Booking-logic tests
|-- requirements.txt  # Python dependencies
|-- .env.example      # Example environment configuration
|-- .gitignore        # Files excluded from Git
`-- README.md
```

## Setup

### 1. Open the project directory

```powershell
cd "C:\Users\HP\Downloads\AI Engineering\Python Learning\dental-appointment-agent"
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file from `.env.example`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit the real `.env` file. It is excluded by `.gitignore`.

## Run the Streamlit interface

```powershell
python -m streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

The page automatically displays all future available appointments without using an LLM call. Enter conversational requests in the chat box below it.

## Deploy on Render

Create a new **Web Service** in Render and connect this GitHub repository.

Use these settings:

```text
Runtime: Python 3
Branch: main
Build Command: pip install -r requirements.txt
Start Command: python -m streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
```

Add this secret environment variable in the Render dashboard:

```text
GROQ_API_KEY=your_real_groq_api_key
```

Do not add the real key to the repository. Render supplies `PORT` automatically, and the start command binds Streamlit to that port so the service can receive public traffic.

Example conversation:

```text
User: Book Ada for 10:00 AM on 2026-12-26.
Assistant: Ada, please confirm your appointment for 2026-12-26 at 10:00 AM.

User: I want to change the time.
Assistant: What would you like to change?

User: Make it 4:30 PM.
Assistant: Your proposed appointment is now 2026-12-26 at 4:30 PM.

User: Yes, confirm it.
Assistant: Your appointment is confirmed.
```

## Run the command-line interface

```powershell
python agent.py
```

Enter `quit` or `exit` to stop it.

## Guardrails

### Explicit confirmation

`set_pending_booking()` creates only a proposal. `book_appointment()` is reached only after the confirmation classifier returns an unambiguous `CONFIRM` result.

### Step limit

The main tool-calling loop stops after five steps to prevent an uncontrolled agent loop.

### Daily call limit

Each in-memory session can attempt at most ten LLM calls per calendar day. Failed and timed-out attempts still count.

### Extraction limit

Natural-language date and time extraction gets at most three attempts. Malformed output cannot create or modify a booking.

### Timeout

Every LLM request has a 60-second timeout. Automatic SDK retries are disabled so one request cannot silently become several minute-long attempts.

### Tool allow-list

The model can request only tools defined in the tool schemas. `execute_tool_call()` checks the requested name and dispatches only approved Python functions.

## Manual scenarios exercised

The complete agent was manually exercised through the command-line interface against the configured model:

- Listed available slots
- Created a proposal
- Confirmed a booking
- Cancelled a proposal without removing its slot
- Modified only the time
- Modified both the date and time
- Rejected an unclear confirmation
- Rejected an unavailable slot
- Handled invalid extraction output
- Reached the daily limit without crashing

## Problems encountered and lessons learned

### 1. A pending proposal was not the same as a booking

The initial flow risked treating a proposed appointment as completed too early. The solution was to store the proposal separately in `session["pending_booking"]` and remove the slot only after explicit confirmation.

### 2. A single-date tool could not answer range questions

`check_available_slots()` could answer questions about one exact date, but the agent could not reliably answer “What is available this week?” It exhausted its step limit because it lacked the correct capability. The solution was to add `list_available_appointments()` for general and range-based queries.

### 3. Deterministic parsing was too restrictive for natural language

A regular expression could extract exact formats such as `2026-12-26` and `4:30 PM`, but it could not reliably interpret phrases such as “tomorrow” or “half past four.” The extractor now uses the LLM to normalize natural language, while Python still validates the result and checks availability.

## Known limitations

- Appointments are stored in memory and reset when the process restarts.
- Confirmed bookings are not associated with persistent patient records.
- The daily usage counter is stored in the current session and resets if the application restarts or a new session is created.
- The schedule is currently hard-coded rather than loaded from a database or calendar service.
- Two users could see and request the same slot because there is no database transaction or concurrency control.
- Authentication is not implemented.
- Natural-language understanding depends on the configured model and provider.

These limitations are intentionally left outside the scope of this Rung 3 project. A production version would use persistent storage, user accounts, transactional booking, monitoring, and a real scheduling system.
