from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Contract(Base):
    __tablename__ = "contracts"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    file_path = Column(String, nullable=True)
    json_path = Column(String, nullable=True)  # New field to store file path for processed markdown sections as JSON
    # faiss_index_path = Column(String, nullable=True)  # New column for FAISS index file path


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    contract_id = Column(Integer, ForeignKey("contracts.id"))
    messages = Column(Text)  # You can store JSON stringified conversation history
    contract = relationship("Contract")