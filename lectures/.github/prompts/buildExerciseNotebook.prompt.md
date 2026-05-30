---
name: buildExerciseNotebook
description: Build exercise notebooks with hidden solutions and detailed explanations.
argument-hint: source=<path-to-md-with-Questions-and-Answers-sections>
---
Create or update the current notebook using a single markdown source file.

Inputs (from the command arguments):
- `source`: required; name of markdown file in `exercises` folder that contains both sections:
  - `## Questions`
  - `## Answers`

Argument format example:
- `source=micrograd.md`

Parsing rules:
- Read questions only from the `## Questions` section.
- Read answers/explanations only from the `## Answers` section.
- If one of these sections is missing, stop and report exactly what is missing.
- In the exercises folder, create a notebook with the same name as the source file (but with a `.ipynb` extension). 

Requirements:
- Preserve every exercise/question text exactly as provided (do not rewrite question wording).
- If needed, map answers even when source notes are out of order.
- For each exercise, keep a visible empty student code cell for learner work.
- Store reference solutions in hidden markdown/text cells (not hidden code cells).
- In each hidden solution cell:
  - include a Python code block with the reference implementation,
  - include a thorough explanation derived from the answer notes,
  - keep this section collapsed/hidden by default.
- Do not expose answers or explanations by default.
- Remove transcript-style trailing citation/reference numbers from exercise prompt text (e.g., ranges like `10-13` or comma refs like `5, 6`) while keeping the instructional sentence intact.
- Keep unrelated notebook content unchanged.

Quality checks:
- Ensure notebook JSON is valid.
- Ensure every exercise has exactly one visible student cell and one hidden solution/explanation markdown cell.
- Ensure no answer content remains in visible non-empty code cells unless explicitly requested.
