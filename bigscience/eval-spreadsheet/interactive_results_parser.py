import clipboard
import json
import time

from parse_eai_results import process_eai_results
from parse_promptsource import process_ps_results

if __name__ == "__main__":
    previous_clipboard = clipboard.paste()
    while True:
        new_clipboard = clipboard.paste()
        if new_clipboard != previous_clipboard:
            previous_clipboard = new_clipboard
            try:
                results_raw = json.loads(new_clipboard)
                if "results" in results_raw:
                    results = process_eai_results(results_raw)
                else:
                    results = process_ps_results(results_raw)
                previous_clipboard = results
            except Exception as e:
                print("🔴 Clipboard content is not a valid EAI results JSON!")

        time.sleep(1)
