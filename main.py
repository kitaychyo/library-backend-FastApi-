from typing import Annotated
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field, EmailStr
import uvicorn
from sqlalchemy import select, Integer, String, UniqueConstraint
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

app = FastAPI()

engine = create_async_engine("sqlite+aiosqlite:///library.db", echo=True)
new_session = async_sessionmaker(engine, expire_on_commit=False)

async def get_session():
    async with new_session() as session:
        yield session

SessionDep = Annotated[AsyncSession, Depends(get_session)]

class Base(DeclarativeBase):
    pass

class BookModel(Base):
    __tablename__ = "books"
    __table_args__ = (UniqueConstraint("isbn"), )

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(nullable=False)
    author: Mapped[str] = mapped_column(nullable=False)
    year: Mapped[int | None] = mapped_column(default=None)
    isbn: Mapped[str | None] = mapped_column(unique=True)
    copies: Mapped[int] = mapped_column(default=1)

class ReaderModel(Base):
    __tablename__ = "readers"
    __table_args__ = (UniqueConstraint("email"), )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    email: Mapped[str] = mapped_column(nullable=False, unique=True)

# ============================
# Schemas
# ============================
class BookBaseSchema(BaseModel):
    title: str
    author: str
    year: int | None = None
    isbn: str | None = None
    copies: int = Field(default=1, ge=0)

class BookCreateSchema(BookBaseSchema):
    pass

class BookSchema(BookBaseSchema):
    id: int
    class Config:
        orm_mode = True

class ReaderBaseSchema(BaseModel):
    name: str
    email: EmailStr

class ReaderCreateSchema(ReaderBaseSchema):
    pass

class ReaderSchema(ReaderBaseSchema):
    id: int
    class Config:
        orm_mode = True


@app.post("/setup")
async def setup_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    return {"ok": True}

# ============================
# CRUD books
# ============================

@app.post("/books", response_model=BookSchema)
async def add_book(data: BookCreateSchema, session: SessionDep):
    new_book = BookModel(**data.dict())
    session.add(new_book)
    try:
        await session.commit()
        await session.refresh(new_book)
        return new_book
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/books", response_model=list[BookSchema])
async def get_books(session: SessionDep):
    result = await session.execute(select(BookModel))
    return result.scalars().all()

@app.get("/books/{book_id}", response_model=BookSchema)
async def get_book(book_id: int, session: SessionDep):
    book = await session.get(BookModel, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@app.put("/books/{book_id}", response_model=BookSchema)
async def update_book(book_id: int, data: BookCreateSchema, session: SessionDep):
    book = await session.get(BookModel, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    for key, value in data.dict().items():
        setattr(book, key, value)
    try:
        await session.commit()
        await session.refresh(book)
        return book
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/books/{book_id}")
async def delete_book(book_id: int, session: SessionDep):
    book = await session.get(BookModel, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    await session.delete(book)
    await session.commit()
    return {"ok": True}

# ============================
# CRUD reader
# ============================
@app.post("/readers", response_model=ReaderSchema)
async def add_reader(data: ReaderCreateSchema, session: SessionDep):
    new_reader = ReaderModel(**data.dict())
    session.add(new_reader)
    try:
        await session.commit()
        await session.refresh(new_reader)
        return new_reader
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/readers", response_model=list[ReaderSchema])
async def get_readers(session: SessionDep):
    result = await session.execute(select(ReaderModel))
    return result.scalars().all()

@app.get("/readers/{reader_id}", response_model=ReaderSchema)
async def get_reader(reader_id: int, session: SessionDep):
    reader = await session.get(ReaderModel, reader_id)
    if not reader:
        raise HTTPException(status_code=404, detail="Reader not found")
    return reader

@app.put("/readers/{reader_id}", response_model=ReaderSchema)
async def update_reader(reader_id: int, data: ReaderCreateSchema, session: SessionDep):
    reader = await session.get(ReaderModel, reader_id)
    if not reader:
        raise HTTPException(status_code=404, detail="Reader not found")
    for key, value in data.dict().items():
        setattr(reader, key, value)
    try:
        await session.commit()
        await session.refresh(reader)
        return reader
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/readers/{reader_id}")
async def delete_reader(reader_id: int, session: SessionDep):
    reader = await session.get(ReaderModel, reader_id)
    if not reader:
        raise HTTPException(status_code=404, detail="Reader not found")
    await session.delete(reader)
    await session.commit()
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
