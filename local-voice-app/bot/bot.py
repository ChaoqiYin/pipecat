#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Compatibility entry point for the shared local voice application."""

import runpy
from pathlib import Path

_app = runpy.run_path(str(Path(__file__).resolve().parents[1] / "bot.py"))
bot = _app["bot"]
run_bot = _app["run_bot"]
run_bot_session = _app["run_bot_session"]
create_stt_service = _app["create_stt_service"]
create_tts_service = _app["create_tts_service"]
create_knowledge_base = _app["create_knowledge_base"]
KnowledgeBase = _app["KnowledgeBase"]
LLMWikiKnowledgeBase = _app["LLMWikiKnowledgeBase"]
create_voice_pipeline = _app["create_voice_pipeline"]
transport_params = _app["transport_params"]

if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
