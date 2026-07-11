# External Method Resource Contract

External methods can be mounted as:
- reference_only: prompt, rubric, protocol, or workflow distilled from a reference repo;
- precomputed: predictions.jsonl loaded as resource evidence;
- shell_command: user-configured command that reads one sample JSON on stdin and emits JSON;
- pipeline_wrapper: an adapter around an existing pipeline that preserves its original prompt, routing, aggregation, and output as much as possible.

Default runtime must not download model weights or require extra scalar reward models.
