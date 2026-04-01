from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.services.artifact_store import ArtifactStore


def test_artifact_store_writes_expected_files(base_settings):
    store = ArtifactStore(base_settings, "artifacts", FileSystemAdapter())
    store.write_json("plan.json", {"hello": "world"})
    store.write_markdown("final_result.md", "# Hi")
    assert (store.root / "plan.json").exists()
    assert (store.root / "final_result.md").exists()
    assert (store.root / "agent_outputs").exists()
