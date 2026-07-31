#!/usr/bin/env python3

from pathlib import Path
import sys
from zipfile import BadZipFile, ZipFile


if len(sys.argv) != 2:
    raise SystemExit("usage: verify_notebook_classpath.py <classpath-dir>")

classpath_dir = Path(sys.argv[1])
required_classes = {
    "com/integrallis/models/api/TextGenerationModel.class",
    "com/integrallis/models/langchain4j/ModelsChatModel.class",
    "com/integrallis/models/spring/ai/ModelsSpringAiChatModel.class",
    "dev/langchain4j/model/chat/ChatModel.class",
    "org/springframework/ai/chat/model/ChatModel.class",
}

available_classes = set()
archives = sorted(classpath_dir.glob("*.jar"))
if not archives:
    raise RuntimeError(f"No JARs found in notebook classpath: {classpath_dir}")

for archive in archives:
    try:
        with ZipFile(archive) as jar:
            available_classes.update(required_classes.intersection(jar.namelist()))
    except BadZipFile as error:
        raise RuntimeError(f"Invalid JAR in notebook classpath: {archive}") from error

missing_classes = sorted(required_classes - available_classes)
if missing_classes:
    formatted = "\n  ".join(missing_classes)
    raise RuntimeError(f"Notebook classpath is missing:\n  {formatted}")

print(f"Verified notebook classpath: {len(archives)} JARs")
