import streamlit as st

from agent import create_session, list_available_appointments, run_agent


st.set_page_config(
    page_title="Dental Appointment Assistant",
    page_icon="🦷",
)

st.title("Dental Appointment Assistant")
st.caption("Check available times and manage a dental appointment.")


availability = list_available_appointments()

with st.expander("Available appointment dates", expanded=True):
    if availability["appointments"]:
        for appointment_date, available_times in availability["appointments"].items():
            formatted_times = ", ".join(available_times)
            st.markdown(f"**{appointment_date}:** {formatted_times}")
    else:
        st.info(availability["message"])


if "agent_session" not in st.session_state:
    st.session_state.agent_session = create_session()

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []


for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_message = st.chat_input("Type your message here...")

if user_message:
    st.session_state.display_messages.append(
        {"role": "user", "content": user_message}
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_message = run_agent(
                user_message,
                st.session_state.agent_session,
            )
        st.markdown(assistant_message)

    st.session_state.display_messages.append(
        {"role": "assistant", "content": assistant_message}
    )
