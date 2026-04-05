"""List all available English edge-tts voices."""
import asyncio
import edge_tts

async def main():
    voices = await edge_tts.list_voices()
    en_voices = [v for v in voices if v["Locale"].startswith("en-")]
    print(f"\n{'Locale':<12} {'ShortName':<45} {'Gender'}")
    print("─" * 70)
    for v in sorted(en_voices, key=lambda x: x["Locale"]):
        print(f"{v['Locale']:<12} {v['ShortName']:<45} {v['Gender']}")

asyncio.run(main())
