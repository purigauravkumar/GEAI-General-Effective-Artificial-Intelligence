from itertools import count
from urllib.parse import urljoin
from fastapi import FastAPI
from ollama import chat
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import json
import os
import re
from backend.memory import *
from backend.crawler import *
from backend.knowledge import *
from backend.projects import *
from backend.filesystem import *
from backend.facts import *
from backend.security import api_key_middleware
from backend import task_engine

app = FastAPI()
app.middleware("http")(api_key_middleware)


def create_project_tool(project_name):

    project_root = WORKSPACE / "Projects" / project_name

    (project_root / "docs").mkdir(parents=True, exist_ok=True)
    (project_root / "notes").mkdir(parents=True, exist_ok=True)
    (project_root / "tasks").mkdir(parents=True, exist_ok=True)
    (project_root / "memory").mkdir(parents=True, exist_ok=True)

    return {
        "action": "create_project",
        "project": project_name,
        "path": str(project_root),
    }


@app.get("/")
def root():
    return {"status": "GEAI ONLINE", "model": "llama3:latest"}


@app.get("/remember")
def remember(text: str):
    memory = load_memory()
    if text not in memory:
        memory.append(text)
    save_memory(memory)
    return {"saved": text}


@app.get("/memory")
def memory():
    return {"memory": load_memory()}


@app.get("/chat")
def geai_chat(prompt: str):
    memory = load_memory()
    memory_text = "\n".join([f"- {item}" for item in memory])
    system_prompt = f"""
You are GEAI.

You have persistent memory.

Current memory:

{memory_text}

Use memory when answering.
"""
    response = chat(
        model="llama3:latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return {"response": response["message"]["content"]}


@app.get("/create_folder")
def create_folder(name: str):
    folder = WORKSPACE / name
    folder.mkdir(parents=True, exist_ok=True)
    return {"created": str(folder)}


@app.get("/create_file")
def create_file(name: str, content: str = ""):
    file_path = WORKSPACE / name
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"created": str(file_path)}


@app.get("/read_file")
def read_file(name: str):
    file_path = WORKSPACE / name
    if not file_path.exists():
        return {"error": "file not found"}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content}


@app.get("/list_workspace")
def list_workspace():
    return {"workspace": [item.name for item in WORKSPACE.iterdir()]}


def command(text: str):
    text_lower = text.lower()

    if text_lower.startswith("test concepts "):
        query = text[14:].strip()
        search_terms = expand_query(query)
        expanded = expand_concepts(search_terms)
        return {"query": query, "search_terms": search_terms, "expanded_terms": expanded}

    if text_lower == "alias list":
        aliases_file = MEMORY_FILE.parent / "aliases.txt"
        if not aliases_file.exists():
            return {"message": "no aliases"}
        with open(aliases_file, "r", encoding="utf-8") as f:
            return {"aliases": f.read()}

    if text_lower.startswith("create project "):
        return create_project(text[15:].strip())
    if text_lower.startswith("create folder "):
        return create_folder(text[14:].strip())
    if text_lower.startswith("create file "):
        remainder = text[12:]
        parts = remainder.split(" with content ", 1)
        return create_file(parts[0].strip(), parts[1] if len(parts) > 1 else "")

    if text_lower.startswith("crawl website "):
        return crawl_website(text[14:].strip())
    if text_lower.startswith("crawl next"):
        return crawl_next()
    if text_lower.startswith("crawl "):
        try:
            return crawl_batch(int(text[6:].strip()))
        except ValueError:
            return {"error": "use: Crawl 10"}
    if text_lower == "rebuild url registry":
        return rebuild_url_registry()
    if text_lower == "upgrade registry":
        return upgrade_registry()
    if text_lower.startswith("recrawl url "):
        return recrawl_url(text[12:].strip())
    if text_lower == "recrawl stale":
        return recrawl_stale()
    if text_lower == "crawler stats":
        return crawler_stats()
    if text_lower == "update freshness":
        return update_freshness_scores()
    if text_lower == "list stale urls":
        return get_stale_urls()
    if text_lower == "freshness report":
        stale = get_stale_urls()
        registry = load_url_registry()
        fresh = len(registry) - stale["count"]
        scores = [item.get("freshness_score", 0) for item in registry.values()]
        return {"known_urls": len(registry), "fresh_urls": fresh, "stale_urls": stale["count"], "average_freshness": round(sum(scores) / len(scores), 2) if scores else 0}
    if text_lower == "knowledge health":
        return knowledge_health()
    if text_lower == "find orphan pages":
        return find_orphan_pages()
    if text_lower == "rebuild index":
        return rebuild_index()
    if text_lower == "rebuild concepts":
        return rebuild_concepts()
    if text_lower == "maintenance report":
        return maintenance_report()
    if text_lower == "registry summary":
        return registry_summary()
    if text_lower == "registry health":
        return registry_health()
    if text_lower == "repair registry":
        return repair_registry()
    if text_lower.startswith("url info "):
        return url_info(text[9:].strip())
    if text_lower.startswith("search report "):
        return search_report(text[14:].strip())
    if text_lower.startswith("knowledge coverage "):
        return knowledge_coverage(text[19:].strip())
    if text_lower.startswith("freshness debug "):
        return freshness_debug(text[17:].strip())
    if text_lower.startswith("explain ranking "):
        return explain_ranking(text[16:].strip())
    if text_lower.startswith("ranking breakdown "):
        return ranking_breakdown(text[18:].strip())
    if text_lower == "ranking health":
        return ranking_health()
    if text_lower.startswith("search quality "):
        return search_quality_report(text[15:].strip())
    if text_lower.startswith("topic authority "):
        return topic_authority(text[16:].strip())
    if text_lower.startswith("topic profile "):
        return topic_profile(text[14:].strip())
    if text_lower.startswith("coverage gaps "):
        return coverage_gaps(text[14:].strip())
    if text_lower == "top topics":
        return top_topics()
    if text_lower.startswith("rank knowledge "):
        return rank_knowledge(text[15:].strip())
    if text_lower.startswith("ask concept "):
        return ask_concept(text[12:].strip())
    if text_lower.startswith("ask knowledge "):
        return ask_knowledge(text[14:].strip())
    if text_lower.startswith("read "):
        return read_file(text[5:].strip())
    if text_lower.startswith("list workspace"):
        return list_workspace()
    if text_lower.startswith("list projects"):
        return list_projects()
    if text_lower.startswith("show project "):
        return show_project(text[13:].strip())
    if text_lower.startswith("create note "):
        remainder = text[12:]
        parts = remainder.split(" in project ", 1)
        if len(parts) < 2:
            return {"error": "use: Create note filename in project ProjectName"}
        return create_note(parts[1].strip(), parts[0].strip())
    if text_lower.startswith("write note "):
        remainder = text[11:]
        parts = remainder.split(" in project ", 1)
        if len(parts) < 2:
            return {"error": "use: Write note file.txt in project ProjectName with content ..."}
        project_parts = parts[1].split(" with content ", 1)
        if len(project_parts) < 2:
            return {"error": "missing content"}
        return write_note(project_parts[0].strip(), parts[0].strip(), project_parts[1])
    if text_lower.startswith("remember in project "):
        remainder = text[len("Remember in project "):]
        words = remainder.split()
        if len(words) < 2:
            return {"error": "use: Remember in project ProjectName MemoryText"}
        return remember_in_project(words[0], " ".join(words[1:]))
    if text_lower.startswith("show memory for project "):
        return show_project_memory(text[24:].strip())
    if text_lower.startswith("search project "):
        return search_project(text[15:].strip())
    if text_lower.startswith("ask project "):
        remainder = text[12:]
        parts = remainder.split(" ", 1)
        if len(parts) < 2:
            return {"error": "use: Ask project ProjectName Question"}
        return ask_project(parts[0].strip(), parts[1].strip())
    if text_lower.startswith("search "):
        return search_all_projects(text[7:].strip().lower())
    if text_lower.startswith("remember fact "):
        parts = text[14:].split(" = ", 1)
        if len(parts) < 2:
            return {"error": "use: Remember fact Topic = Fact"}
        return remember_fact(parts[0].strip(), parts[1].strip())
    if text_lower.startswith("remember "):
        memory_item = text[9:].strip()
        memory = load_memory()
        if memory_item not in memory:
            memory.append(memory_item)
        save_memory(memory)
        return {"action": "remember", "saved": memory_item}
    if text_lower.startswith("ask fact "):
        return ask_fact(text[9:].strip())
    return {"action": "unknown", "message": "command not recognized"}


def route_prompt(prompt: str):
    response = chat(
        model="llama3:latest",
        messages=[
            {"role": "system", "content": """
You are a router.
Reply with ONLY ONE WORD.
CHAT
MEMORY
PROJECT
Examples:
What is our roadmap? -> PROJECT
What do we know about GEAI? -> PROJECT
Search memory -> MEMORY
Remember this -> MEMORY
Tell me a joke -> CHAT
Who are you? -> CHAT
"""},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"].strip().upper()


@app.get("/command")
def command_endpoint(text: str):
    return command(text)


@app.get("/ask")
def ask(prompt: str):
    route = route_prompt(prompt)
    if route == "PROJECT":
        return command("Ask project GEAI " + prompt)
    if route == "MEMORY":
        return memory()
    return geai_chat(prompt)


# GEAI Task Queue API
# Mutating operations use POST so they are not accidentally triggered by link previews or crawlers.
from typing import Any
from pydantic import BaseModel
from fastapi import HTTPException, Query


class CreateTaskRequest(BaseModel):
    title: str
    description: str = ""


class CompleteTaskRequest(BaseModel):
    result: Any = None


class FailTaskRequest(BaseModel):
    error: str


@app.post("/tasks")
def api_create_task(payload: CreateTaskRequest):
    try:
        return task_engine.create_task(payload.title, payload.description)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/tasks")
def api_list_tasks(
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    try:
        return {"tasks": task_engine.list_tasks(status=status, limit=limit)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/tasks/{task_id}")
def api_get_task(task_id: str):
    task = task_engine.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.post("/tasks/claim-next")
def api_claim_next_task():
    return {"task": task_engine.claim_next_task()}


@app.post("/tasks/{task_id}/complete")
def api_complete_task(task_id: str, payload: CompleteTaskRequest):
    try:
        return task_engine.complete_task(task_id, payload.result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/tasks/{task_id}/fail")
def api_fail_task(task_id: str, payload: FailTaskRequest):
    try:
        return task_engine.fail_task(task_id, payload.error)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/tasks/{task_id}/cancel")
def api_cancel_task(task_id: str):
    try:
        return task_engine.cancel_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
