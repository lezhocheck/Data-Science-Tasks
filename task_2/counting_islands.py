# map can be reprented as undirected graph
# number of islands = number of unconnected components of such graph
def solution() -> None:
    rows, columns = to_list(input())
    map_data = []
    for _ in range(rows):
        row = to_list(input())
        assert len(row) == columns
        map_data.append(row)
    map = Map(map_data)
    result = map.calculate_islands()
    print(result) 


def to_list(value: str, delimiter: str = ' ') -> list[int]:
    return [int(i) for i in value.split(delimiter) if i]


class Map:
    def __init__(self, data: list[list[int]]) -> None:
        self._graph = data
        self._rows = len(self._graph)
        self._columns = len(self._graph[0]) if self._rows > 0 else 0

    def calculate_islands(self) -> int:
        result = 0
        for row in range(self._rows):
            for column in range(self._columns):
                if self._graph[row][column] == 1:
                    self._dfs(row, column)
                    result += 1
        return result
    
    def _dfs(self, row: int, column: int) -> None:
        if row < 0 or column < 0 or row >= self._rows or column >= self._columns:
            return
        if self._graph[row][column] != 1:
            return
        self._graph[row][column] = -1
        # check 4 neighbors
        self._dfs(row - 1, column)
        self._dfs(row + 1, column)
        self._dfs(row, column - 1)
        self._dfs(row, column + 1)


if __name__ == '__main__':
    solution()
