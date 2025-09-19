from __future__ import annotations

import os
from . import (
    PolliClient,
    generate_text,
    save_image_timestamped,
    chat_completion,
    chat_completion_stream,
    chat_completion_tools,
    analyze_image_url,
    transcribe_audio,
    image_feed_stream,
    text_feed_stream,
)


def main() -> None:
    c = PolliClient()
    print("Text models:", len(c.list_models("text")))
    print("Image models:", len(c.list_models("image")))
    for q in ("flux", "openai", "gemini", "nonexistent"):
        hit = c.get_model_by_name(q)
        print(f"{q!r} ->", hit["name"] if hit else None)

    print("\n--- Text Generation Example ---")
    txt = generate_text("Explain the theory of relativity simply")
    print(txt)

    print("\n--- Image Generation Example ---")
    saved = save_image_timestamped("A beautiful sunset over the ocean")
    print("Image saved to:", saved)

    print("\n--- Chat Completion Example ---")
    msgs = [
        {"role": "system", "content": "You are a helpful historian."},
        {"role": "user", "content": "When did the French Revolution start?"},
    ]
    reply = chat_completion(msgs)
    print("Assistant:", reply)

    print("\n--- Chat Completion Streaming Example ---")
    stream_msgs = [
        {"role": "user", "content": "Tell me a story that unfolds slowly."}
    ]
    for part in chat_completion_stream(stream_msgs):
        print(part, end="", flush=True)
    print("\n[stream complete]")

    print("\n--- Function Calling Example ---")
    def get_current_weather(location: str, unit: str = "celsius"):
        if "tokyo" in (location or "").lower():
            return {"location": location, "temperature": "15", "unit": unit, "description": "Cloudy"}
        return {"location": location, "temperature": "unknown"}

    tool_spec = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    fc_messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    fc_reply = chat_completion_tools(fc_messages, tools=tool_spec, functions={"get_current_weather": get_current_weather})
    print("Assistant:", fc_reply)

    print("\n--- Vision (URL) Example ---")
    vision_reply = analyze_image_url(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/1024px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        question="Describe the main subject.",
    )
    print(vision_reply)

    try:
        sample = os.path.join(os.getcwd(), "sample.wav")
        if os.path.exists(sample):
            print("\n--- Speech-to-Text Example ---")
            transcript = transcribe_audio(sample)
            print(transcript)
        else:
            print("\n[Skipping Speech-to-Text; no sample.wav found]")
    except Exception as e:
        print("\n[Speech-to-Text error]", e)

    # Real-time public feeds (endless). Uncomment to run.
    # print("\n--- Public Image Feed (endless) ---")
    # for event in image_feed_stream(reconnect=True, include_bytes=True):
    #     print("New image:", event.get("prompt"), len(event.get("image_bytes") or b""), "bytes")
    #     # break

    # print("\n--- Public Text Feed (endless) ---")
    # for event in text_feed_stream(reconnect=True):
    #     print("Model:", event.get("model"))
    #     msgs = event.get("messages") or []
    #     user = next((m for m in msgs if m.get("role") == "user"), None)
    #     if user and user.get("content"):
    #         content = user["content"]
    #         preview = content if isinstance(content, str) else str(content)
    #         print("User:", (preview[:80] + ("..." if len(preview) > 80 else "")))
    #     print("Response:", (event.get("response", "")[:100] + "..."))


if __name__ == "__main__":
    main()

