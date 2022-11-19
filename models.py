from pydantic import BaseModel

class Coordinate(BaseModel):
    coordinates: list[list[list[float]]]