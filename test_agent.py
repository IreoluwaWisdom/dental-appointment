import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent


class BookingLogicTests(unittest.TestCase):
    def setUp(self):
        self.original_appointments = {
            appointment_date: slots.copy()
            for appointment_date, slots in agent.appointments.items()
        }

    def tearDown(self):
        agent.appointments.clear()
        agent.appointments.update(self.original_appointments)

    def test_pending_booking_does_not_remove_slot(self):
        session = agent.create_session()
        result = agent.set_pending_booking(session, "2026-12-26", "2:00 PM", "John")
        self.assertTrue(result["success"])
        self.assertIn("2:00 PM", agent.check_available_slots("2026-12-26"))

    def test_confirmed_booking_removes_slot(self):
        session = agent.create_session()
        agent.set_pending_booking(session, "2026-12-26", "2:00 PM", "John")
        result = agent.process_booking_confirmation(session, True)
        self.assertTrue(result["success"])
        self.assertNotIn("2:00 PM", agent.check_available_slots("2026-12-26"))
        self.assertIsNone(session["pending_booking"])

    def test_cancelled_booking_keeps_slot(self):
        session = agent.create_session()
        agent.set_pending_booking(session, "2026-12-26", "2:00 PM", "John")
        result = agent.process_booking_confirmation(session, False)
        self.assertFalse(result["success"])
        self.assertIn("2:00 PM", agent.check_available_slots("2026-12-26"))

    def test_modification_can_change_only_time(self):
        session = agent.create_session()
        agent.set_pending_booking(session, "2026-12-26", "2:00 PM", "John")
        result = agent.modify_pending_booking(session, new_time="4:30 PM")
        self.assertTrue(result["success"])
        self.assertEqual(session["pending_booking"]["date"], "2026-12-26")
        self.assertEqual(session["pending_booking"]["time"], "4:30 PM")

    @patch("agent.extract_booking_changes", return_value=(None, "4:30 PM"))
    @patch("agent.classify_booking_response", return_value="MODIFY")
    def test_modification_conversation_has_two_steps(
        self, _mock_classifier, _mock_extractor
    ):
        session = agent.create_session()
        agent.set_pending_booking(session, "2026-12-26", "2:00 PM", "John")
        first_result = agent.handle_booking_response("Can I change it?", session)
        second_result = agent.handle_booking_response("Make it 4:30 PM", session)
        self.assertTrue(first_result["success"])
        self.assertTrue(second_result["success"])
        self.assertEqual(session["pending_booking"]["time"], "4:30 PM")
        self.assertFalse(session["awaiting_modification"])

    def test_unavailable_modification_does_not_change_proposal(self):
        session = agent.create_session()
        agent.set_pending_booking(session, "2026-12-26", "2:00 PM", "John")
        result = agent.modify_pending_booking(session, new_time="3:00 PM")
        self.assertFalse(result["success"])
        self.assertEqual(session["pending_booking"]["time"], "2:00 PM")

    @patch("agent.client.chat.completions.create")
    def test_llm_extracts_natural_language_booking_changes(self, mock_create):
        session = agent.create_session()
        mock_create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"date": "2026-12-26", "time": "4:30 PM"}'
                    )
                )
            ]
        )

        new_date, new_time = agent.extract_booking_changes(
            "Move my appointment to December 26 at half past four in the afternoon.",
            session,
        )

        self.assertEqual(new_date, "2026-12-26")
        self.assertEqual(new_time, "4:30 PM")
        self.assertEqual(session["llm_call_count"], 1)
        mock_create.assert_called_once()

    @patch("agent.client.chat.completions.create")
    def test_invalid_extraction_output_fails_safely(self, mock_create):
        session = agent.create_session()
        mock_create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))]
        )

        new_date, new_time = agent.extract_booking_changes(
            "Change the booking.",
            session,
        )

        self.assertIsNone(new_date)
        self.assertIsNone(new_time)
        self.assertEqual(session["llm_call_count"], agent.MAX_EXTRACTION_ATTEMPTS)
        self.assertEqual(mock_create.call_count, agent.MAX_EXTRACTION_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
