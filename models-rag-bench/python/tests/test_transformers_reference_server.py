import transformers_reference_server as reference
import pytest


def test_stop_sequences_are_applied_before_streaming_text():
    visible, stopped = reference.apply_stop_sequences("answer\n\nignored", ("\n\n",))

    assert visible == "answer"
    assert stopped


def test_completion_events_match_the_java_raw_completion_parser():
    token = reference.content_event("route")
    final = reference.final_event(prompt_tokens=12, output_tokens=3, prompt_millis=45.5)

    assert token == {"content": "route"}
    assert final == {
        "content": "",
        "stop": True,
        "timings": {"prompt_n": 12, "predicted_n": 3, "prompt_ms": 45.5},
    }


def test_accepts_the_controlled_benchmark_greedy_settings():
    reference.validate_greedy_controls(
        {
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "repeat_penalty": 1,
            "cache_prompt": False,
        }
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("temperature", 0.5),
        ("top_k", 40),
        ("top_p", 0.9),
        ("repeat_penalty", 1.1),
        ("cache_prompt", True),
    ],
)
def test_rejects_controls_the_reference_does_not_implement(name, value):
    request = {
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "repeat_penalty": 1,
        "cache_prompt": False,
    }
    request[name] = value

    with pytest.raises(ValueError, match="deterministic controls"):
        reference.validate_greedy_controls(request)
