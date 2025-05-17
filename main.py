from typing import Annotated
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, EmailStr
import uvicorn
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

app = FastAPI()

engine = create_async_engine('sqlite+aiosqlite:///library.db', echo=True)
new_session = async_sessionmaker(engine, expire_on_commit=False)

async def get_session():
    async with new_session() as session:
        yield session

SessionDep = Annotated[AsyncSession, Depends(get_session)]

class Base(DeclarativeBase):
    pass
# -------------------- Модели ------------------------
class BookModel(Base):
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str]
    author: Mapped[str]
    year: Mapped[int | None]
    isbn: Mapped[str | None] = mapped_column(unique=True)
    copies: Mapped[int]

class ReaderModel(Base):
    __tablename__ = "readers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str]
    email: Mapped[str]

# -------------------- Это лишнее ------------------------
@app.post("/setup",
          tags=['Полный сброс бд']
          )
async def setup_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    return {"message": "База данных и таблицы созданы заново"}


# -------------------- Схема книги ------------------------
class BookAddSchema(BaseModel):
    title: str
    author: str
    year: int = Field(ge=0)
    isbn: str | None = None
    copies: int = Field(ge=0, default=1)

class BookSchema(BookAddSchema):
    id: int

# -------------------- Схема читателя ------------------------
class ReaderAddSchema(BaseModel):
    name: str
    email: EmailStr

class ReaderSchema(ReaderAddSchema):
    id: int

# -------------------- Книги ------------------------

@app.post("/add_books", response_model=BookSchema,tags=['Книги'])
async def add_book(data: BookAddSchema, session: SessionDep):
    new_book = BookModel(**data.dict())
    session.add(new_book)
    await session.commit()
    await session.refresh(new_book)
    return new_book

@app.get("/view_books", response_model=list[BookSchema], tags=['Книги'])
async def get_books(session: SessionDep):
    result = await session.execute(select(BookModel))
    return result.scalars().all()

@app.get("/view_book/{book_id}", response_model=BookSchema, tags=['Книги'])
async def get_book(book_id: int, session: SessionDep):
    book = await session.get(BookModel, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")
    return book

@app.put("/update_book/{book_id}", response_model=BookSchema, tags=['Книги'])
async def update_book(book_id: int, data: BookAddSchema, session: SessionDep):
    book = await session.get(BookModel, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")

    for key, value in data.dict().items():
        setattr(book, key, value)

    await session.commit()
    await session.refresh(book)
    return book

@app.delete("/delete_books/{book_id}", tags=['Книги'])
async def delete_book(book_id: int, session: SessionDep):
    book = await session.get(BookModel, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")

    await session.delete(book)
    await session.commit()
    return {"ok": True, "message": "Книга удалена"}


# -------------------- Читатели ------------------------

@app.post("/add_reader", response_model=ReaderSchema, tags=['Читатели'])
async def add_reader(data: ReaderAddSchema, session: SessionDep):
    new_reader = ReaderModel(**data.dict())
    session.add(new_reader)
    await session.commit()
    await session.refresh(new_reader)
    return new_reader

@app.get("/view_readers", response_model=list[ReaderSchema], tags=['Читатели'])
async def get_readers(session: SessionDep):
    result = await session.execute(select(ReaderModel))
    return result.scalars().all()

@app.get("/view_reader/{reader_id}", response_model=ReaderSchema, tags=['Читатели'])
async def get_reader(reader_id: int, session: SessionDep):
    reader = await session.get(ReaderModel, reader_id)
    if not reader:
        raise HTTPException(status_code=404, detail="Читатель не найден")
    return reader

@app.put("/update_reader/{reader_id}", response_model=ReaderSchema, tags=['Читатели'])
async def update_reader(reader_id: int, data: ReaderAddSchema, session: SessionDep):
    reader = await session.get(ReaderModel, reader_id)
    if not reader:
        raise HTTPException(status_code=404, detail="Читатель не найден")

    for key, value in data.dict().items():
        setattr(reader, key, value)

    await session.commit()
    await session.refresh(reader)
    return reader

@app.delete("/delete_reader/{reader_id}", tags=['Читатели'])
async def delete_reader(reader_id: int, session: SessionDep):
    book = await session.get(ReaderModel, reader_id)
    if not book:
        raise HTTPException(status_code=404, detail="Читатель не найден")

    await session.delete(book)
    await session.commit()
    return {"ok": True, "message": "Читатель удален"}


# Запуск
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
