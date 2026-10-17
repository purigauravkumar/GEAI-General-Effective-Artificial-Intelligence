import pytest

from backend import task_engine as tasks


def test_task_lifecycle_is_persistent_and_records_result(tmp_path, monkeypatch):
    monkeypatch.setenv("GEAI_HOME", str(tmp_path))
    created = tasks.create_task("  Build GEAI  ", "First executable milestone")
    assert created["title"] == "Build GEAI"
    assert created["status"] == "queued"
    assert tasks.get_task(created["id"])["status"] == "queued"

    claimed = tasks.claim_next_task()
    assert claimed["id"] == created["id"]
    assert claimed["status"] == "running"
    assert claimed["attempts"] == 1

    completed = tasks.complete_task(created["id"], {"tests": 4, "passed": True})
    assert completed["status"] == "succeeded"
    assert completed["result"] == {"tests": 4, "passed": True}
    assert tasks.claim_next_task() is None


def test_empty_title_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GEAI_HOME", str(tmp_path))
    with pytest.raises(ValueError, match="title"):
        tasks.create_task("   ")


def test_fail_and_cancel_transitions(tmp_path, monkeypatch):
    monkeypatch.setenv("GEAI_HOME", str(tmp_path))
    failed_task = tasks.create_task("Fail me")
    tasks.claim_next_task()
    failed = tasks.fail_task(failed_task["id"], "provider unavailable")
    assert failed["status"] == "failed"
    assert failed["error"] == "provider unavailable"

    queued = tasks.create_task("Cancel me")
    cancelled = tasks.cancel_task(queued["id"])
    assert cancelled["status"] == "cancelled"
    assert tasks.list_tasks(status="cancelled")[0]["id"] == queued["id"]


def test_invalid_state_transitions_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GEAI_HOME", str(tmp_path))
    created = tasks.create_task("Not claimed")
    with pytest.raises(ValueError, match="cannot finish"):
        tasks.complete_task(created["id"], "unexpected")
    tasks.claim_next_task()
    with pytest.raises(ValueError, match="only queued"):
        tasks.cancel_task(created["id"])


def test_unknown_task_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GEAI_HOME", str(tmp_path))
    with pytest.raises(KeyError, match="task not found"):
        tasks.complete_task("does-not-exist")
