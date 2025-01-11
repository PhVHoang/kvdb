import mmap
import struct
import threading
import hashlib
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Tuple
from pathlib import Path

# Constants
NUM_LANES = 64
NUM_SHARDS = 1024
PAGE_SIZE = 4096
FIRST_LANE_PAGES = 64

K = TypeVar('K')
V = TypeVar('V')

# Global locks for shards
SHARD_LOCKS = [threading.Lock() for _ in range(NUM_SHARDS)]


@dataclass
class Entry(Generic[K, V]):
    key: K
    val: V
    next: int
    kv_checksum: int
    next_checksum: int

    @staticmethod
    def hash_val(value) -> int:
        """Calculate hash for a value."""
        return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], 'little')

    def valid(self) -> bool:
        """Check if entry is valid based on checksums."""
        return (self.hash_val(self.key) + self.hash_val(self.val) == self.kv_checksum and
                self.next + 1 == self.next_checksum)

    def set_next(self, next_val: int) -> None:
        """Set the next pointer and update its checksum."""
        self.next = next_val
        self.next_checksum = next_val + 1

    @classmethod
    def new(cls, key: K, val: V) -> 'Entry[K, V]':
        """Create a new entry with proper checksums."""
        kv_checksum = cls.hash_val(key) + cls.hash_val(val)
        entry = cls(key, val, 0, kv_checksum, 1)
        assert entry.valid()
        return entry

    def pack(self) -> bytes:
        """Pack entry into bytes for storage."""
        format_str = 'qqdqq'  # Assuming key and value are integers for simplicity
        return struct.pack(format_str, self.key, self.val, self.next,
                           self.kv_checksum, self.next_checksum)

    @classmethod
    def unpack(cls, data: bytes) -> 'Entry[K, V]':
        """Unpack entry from bytes."""
        key, val, next_val, kv_checksum, next_checksum = struct.unpack('qqdqq', data)
        return cls(key, val, next_val, kv_checksum, next_checksum)


class Index(Generic[K, V]):
    """On-disk index structure mapping keys to values."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lanes = []
        self.pages_lock = threading.Lock()
        self._num_pages = 0

        # Initialize or load existing lanes
        self._init_lanes()

        # Ensure at least one page exists
        if self._num_pages == 0:
            self.new_page()

    def _init_lanes(self) -> None:
        """Initialize or load existing lanes from disk."""
        for n in range(NUM_LANES):
            lane_path = self.path / f"{n:02x}"

            if lane_path.exists():
                with open(lane_path, 'r+b') as f:
                    lane_pages = self._lane_pages(n)
                    f.truncate(PAGE_SIZE * lane_pages)
                    mm = mmap.mmap(f.fileno(), 0)
                    self.lanes.append(mm)
                    self._num_pages = max(self._num_pages,
                                          self._count_pages_in_lane(mm, n))

    def _count_pages_in_lane(self, mm: mmap.mmap, lane_num: int) -> int:
        """Count number of valid pages in a lane."""
        entry_size = struct.calcsize('qqdqq')
        entries_per_page = PAGE_SIZE // entry_size

        total_pages = 0
        for page in range(self._lane_pages(lane_num)):
            for slot in range(entries_per_page):
                offset = page * PAGE_SIZE + slot * entry_size
                entry_data = mm[offset:offset + entry_size]
                if not entry_data:
                    continue
                try:
                    entry = Entry.unpack(entry_data)
                    if entry.valid():
                        total_pages = max(total_pages, page + 1)
                except struct.error:
                    continue
        return total_pages

    @staticmethod
    def _lane_pages(n: int) -> int:
        """Calculate number of pages in a lane."""
        return 2 ** n * FIRST_LANE_PAGES

    @staticmethod
    def _lane_page(page: int) -> Tuple[int, int]:
        """Calculate lane and page offset for a given page number."""
        i = page // FIRST_LANE_PAGES + 1
        lane = (i - 1).bit_length() - 1
        page_offset = page - (2 ** lane - 1) * FIRST_LANE_PAGES
        return lane, page_offset

    def new_lane(self) -> None:
        """Create a new lane file."""
        lane_num = len(self.lanes)
        lane_path = self.path / f"{lane_num:02x}"
        num_pages = self._lane_pages(lane_num)

        with open(lane_path, 'wb') as f:
            f.truncate(PAGE_SIZE * num_pages)

        with open(lane_path, 'r+b') as f:
            self.lanes.append(mmap.mmap(f.fileno(), 0))

    def new_page(self) -> int:
        """Allocate a new page."""
        with self.pages_lock:
            lane, offset = self._lane_page(self._num_pages)

            if offset == 0:
                self.new_lane()

            new_page = self._num_pages
            self._num_pages += 1
            return new_page

    def _get_entry(self, lane: int, page: int, slot: int) -> Optional[Entry]:
        """Get entry at specified location."""
        if lane >= len(self.lanes):
            return None

        entry_size = struct.calcsize('qqdqq')
        offset = page * PAGE_SIZE + slot * entry_size

        try:
            entry_data = self.lanes[lane][offset:offset + entry_size]
            return Entry.unpack(entry_data)
        except (IndexError, struct.error):
            return None

    def _set_entry(self, lane: int, page: int, slot: int, entry: Entry) -> None:
        """Set entry at specified location."""
        entry_size = struct.calcsize('qqdqq')
        offset = page * PAGE_SIZE + slot * entry_size
        self.lanes[lane][offset:offset + entry_size] = entry.pack()

    def insert(self, key: K, val: V) -> bool:
        """Insert a key-value pair into the index."""
        entry = Entry.new(key, val)
        hash_val = Entry.hash_val(key)
        depth = 0

        while True:
            slot = hash_val % (PAGE_SIZE // struct.calcsize('qqdqq'))
            lane, page = self._lane_page(depth)

            shard = (page ^ slot) % NUM_SHARDS
            with SHARD_LOCKS[shard]:
                current = self._get_entry(lane, page, slot)

                if not current or not current.valid():
                    self._set_entry(lane, page, slot, entry)
                    return True
                elif current.key == key:
                    return False
                elif current.next == 0:
                    current.set_next(self.new_page())
                    self._set_entry(lane, page, slot, current)

                depth = current.next

    def get(self, key: K) -> Optional[V]:
        """Look up a value by key."""
        hash_val = Entry.hash_val(key)
        depth = 0

        while True:
            slot = hash_val % (PAGE_SIZE // struct.calcsize('qqdqq'))
            lane, page = self._lane_page(depth)

            entry = self._get_entry(lane, page, slot)
            if not entry or not entry.valid():
                return None

            if entry.key == key:
                return entry.val
            elif entry.next == 0:
                return None

            depth = entry.next

    def flush(self) -> None:
        """Synchronize and flush data to disk."""
        for lane in self.lanes:
            lane.flush()

    def close(self) -> None:
        """Close all memory maps."""
        for lane in self.lanes:
            lane.close()
