from abc import ABC, abstractmethod

class BaseDatabaseRepository(ABC):
    
    @abstractmethod
    async def save_product_analysis(self, payload: dict) -> bool:
        """Saves product dimension, category, and pricing results."""
        pass

    @abstractmethod
    async def get_all_categories(self) -> list:
        """Fetches list of active product categories."""
        pass

# In-Memory / Local JSON Fallback Repository
class DummyRepository(BaseDatabaseRepository):
    def __init__(self):
        self.records = []

    async def save_product_analysis(self, payload: dict) -> bool:
        self.records.append(payload)
        return True

    async def get_all_categories(self) -> list:
        return ["necklace", "bracelet", "basket", "tray", "pottery"]
