from agent.services.task_parser import parse_task_items


def test_parse_incomplete_checkbox_task() -> None:
    tasks = parse_task_items("- [ ] Create simulation")

    assert [(task.text, task.state) for task in tasks] == [
        ("Create simulation", "incomplete")
    ]


def test_parse_complete_checkbox_tasks() -> None:
    tasks = parse_task_items("- [x] Rename folder\n* [X] Create overview")

    assert [(task.text, task.state) for task in tasks] == [
        ("Rename folder", "complete"),
        ("Create overview", "complete"),
    ]


def test_parse_plain_bullet_as_unknown() -> None:
    tasks = parse_task_items("+ Write model documentation")

    assert [(task.text, task.state) for task in tasks] == [
        ("Write model documentation", "unknown")
    ]
