from recipe_dapt.data import MixedPackedDataset


class Rows:
    def __init__(self, prefix: str, count: int) -> None:
        self.rows = [{"input_ids": [prefix, index]} for index in range(count)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def test_periodic_five_percent_replay() -> None:
    mixed = MixedPackedDataset(Rows("recipe", 100), Rows("replay", 10), 0.05, 40)
    sources = [mixed[index]["input_ids"][0] for index in range(40)]
    assert sources.count("replay") == 2
    assert sources[0] == "replay"
    assert sources[20] == "replay"


def test_recipe_only_view() -> None:
    mixed = MixedPackedDataset(Rows("recipe", 3), None, 0.0, 5)
    assert [mixed[index]["input_ids"][1] for index in range(5)] == [0, 1, 2, 0, 1]
