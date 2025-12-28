from abc import ABC, abstractmethod

class SoftwareAgent(ABC):
    """
    Osnovna apstrakcija za svakog agenta. 
    Prati princip Sense -> Think -> Act.
    """

    @abstractmethod
    def step(self):
        """
        Izvršava jednu atomarnu iteraciju agenta (jedan 'tick').
        Vraća TickResult ako je bilo posla, ili None ako nema šta da radi.
        """
        pass


class TickResult:
    """
    Standardizovani izlaz svakog koraka agenta.
    Omogućava Web sloju da zna šta se desilo bez poznavanja domenske logike.
    """
    def __init__(self, item_id, probability, decision, status):
        self.item_id = item_id          # ID entiteta koji je obrađen
        self.probability = probability  # Rezultat ML modela (0.0 - 1.0)
        self.decision = decision        # "Call", "Ignore", "Pending" itd.
        self.status = status            # Status akcije (npr. "Processed", "Skipped")

    def to_dict(self):
        """Pomoćna metoda za lakši prikaz u Web sloju/JSON-u."""
        return {
            "item_id": self.item_id,
            "probability": self.probability,
            "decision": self.decision,
            "status": self.status
        }