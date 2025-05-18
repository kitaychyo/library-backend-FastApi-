
from typing import Annotated, List
from fastapi import FastAPI, Depends, HTTPException, status, Response, Request
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordRequestForm
import jwt

# Инициализация FastAPI приложения
app = FastAPI(title="Library API", description="API для управления библиотекой")

# Настройка базы данных SQLite
engine = create_async_engine("sqlite+aiosqlite:///library.db", echo=True)
new_session = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    async with new_session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]

# Настройка JWT и хеширования паролей
JWT_SECRET_KEY = "your-secret-key"  # Замените на безопасный ключ в продакшене
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


# Базовый класс для модели SQLAlchemy
class Base(DeclarativeBase):
    pass


# Модель пользователя (библиотекаря)
class UserModel(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(unique=True)
    hashed_password: Mapped[str]


# Модель книги
class BookModel(Base):
    __tablename__ = "books"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str]
    author: Mapped[str]
    year: Mapped[int | None]
    isbn: Mapped[str | None] = mapped_column(unique=True)
    copies: Mapped[int]
    description: Mapped[str | None]


# Модель читателя
class ReaderModel(Base):
    __tablename__ = "readers"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)


# Модель записи о выдаче книги
class BorrowedBookModel(Base):
    __tablename__ = "borrowed_books"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id"))
    reader_id: Mapped[int] = mapped_column(ForeignKey("readers.id"))
    borrow_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    return_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# Pydantic схемы для валидации данных
class UserRegisterSchema(BaseModel):
    email: EmailStr
    password: str


class UserSchema(BaseModel):
    id: int
    email: EmailStr


class BookAddSchema(BaseModel):
    title: str
    author: str
    year: int | None = Field(ge=0, default=None)
    isbn: str | None = None
    copies: int = Field(ge=0, default=1)
    description: str | None = None


class BookSchema(BookAddSchema):
    id: int


class ReaderAddSchema(BaseModel):
    name: str
    email: EmailStr


class ReaderSchema(ReaderAddSchema):
    id: int


class BorrowSchema(BaseModel):
    book_id: int
    reader_id: int


class BorrowedBookSchema(BaseModel):
    id: int
    book_id: int
    reader_id: int
    borrow_date: datetime
    return_date: datetime | None


# Функция для создания JWT-токена
def create_access_token(subject: str) -> str:
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode = {"sub": subject, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


# Функция для проверки токена из cookie
async def get_current_user(request: Request, session: SessionDep) -> UserSchema:
    try:
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")

        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        result = await session.execute(select(UserModel).filter_by(id=int(user_id)))
        user = result.scalars().first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")

        return UserSchema(id=user.id, email=user.email)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


UserDep = Annotated[UserSchema, Depends(get_current_user)]


# Эндпоинты аутентификации
@app.post("/register", response_model=UserSchema, tags=["Аутентификация"])
async def register_user(data: UserRegisterSchema, session: SessionDep):
    result = await session.execute(select(UserModel).filter_by(email=data.email))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email уже зарегистрирован")

    hashed_password = pwd_context.hash(data.password)
    user = UserModel(email=data.email, hashed_password=hashed_password)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@app.post("/login", tags=["Аутентификация"])
async def login_user(session: SessionDep, response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    result = await session.execute(select(UserModel).filter_by(email=form_data.username))
    user = result.scalars().first()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    token = create_access_token(str(user.id))
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,  # Защита от XSS
        secure=False,  # Установите True в продакшене с HTTPS
        samesite="none",  # Без CSRF
        max_age=7 * 24 * 60 * 60  # 7 дней в секундах
    )
    return {"message": "Успешный вход"}


# Эндпоинты для управления книгами
@app.post("/add_books", response_model=BookSchema, tags=["Книги"])
async def add_book(data: BookAddSchema, session: SessionDep, user: UserDep):
    new_book = BookModel(**data.dict())
    session.add(new_book)
    await session.commit()
    await session.refresh(new_book)
    return new_book


@app.get("/view_books", response_model=List[BookSchema], tags=["Книги"])
async def get_books(session: SessionDep):
    result = await session.execute(select(BookModel))
    return result.scalars().all()


@app.get("/view_book/{book_id}", response_model=BookSchema, tags=["Книги"])
async def get_book(book_id: int, session: SessionDep, user: UserDep):
    book = await session.get(BookModel, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")
    return book


@app.put("/update_book/{book_id}", response_model=BookSchema, tags=["Книги"])
async def update_book(book_id: int, data: BookAddSchema, session: SessionDep, user: UserDep):
    book = await session.get(BookModel, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")

    for key, value in data.dict().items():
        setattr(book, key, value)

    await session.commit()
    await session.refresh(book)
    return book


@app.delete("/delete_books/{book_id}", tags=["Книги"])
async def delete_book(book_id: int, session: SessionDep, user: UserDep):
    book = await session.get(BookModel, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")

    await session.delete(book)
    await session.commit()
    return {"ok": True, "message": "Книга удалена"}


# Эндпоинты для управления читателями
@app.post("/add_reader", response_model=ReaderSchema, tags=["Читатели"])
async def add_reader(data: ReaderAddSchema, session: SessionDep, user: UserDep):
    result = await session.execute(select(ReaderModel).filter_by(email=data.email))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email уже зарегистрирован")

    new_reader = ReaderModel(**data.dict())
    session.add(new_reader)
    await session.commit()
    await session.refresh(new_reader)
    return new_reader


@app.get("/view_readers", response_model=List[ReaderSchema], tags=["Читатели"])
async def get_readers(session: SessionDep, user: UserDep):
    result = await session.execute(select(ReaderModel))
    return result.scalars().all()


@app.get("/view_reader/{reader_id}", response_model=ReaderSchema, tags=["Читатели"])
async def get_reader(reader_id: int, session: SessionDep, user: UserDep):
    reader = await session.get(ReaderModel, reader_id)
    if not reader:
        raise HTTPException(status_code=404, detail="Читатель не найден")
    return reader


@app.put("/update_reader/{reader_id}", response_model=ReaderSchema, tags=["Читатели"])
async def update_reader(reader_id: int, data: ReaderAddSchema, session: SessionDep, user: UserDep):
    reader = await session.get(ReaderModel, reader_id)
    if not reader:
        raise HTTPException(status_code=404, detail="Читатель не найден")

    for key, value in data.dict().items():
        setattr(reader, key, value)

    await session.commit()
    await session.refresh(reader)
    return reader


@app.delete("/delete_reader/{reader_id}", tags=["Читатели"])
async def delete_reader(reader_id: int, session: SessionDep, user: UserDep):
    reader = await session.get(ReaderModel, reader_id)
    if not reader:
        raise HTTPException(status_code=404, detail="Читатель не найден")

    await session.delete(reader)
    await session.commit()
    return {"ok": True, "message": "Читатель удален"}


# Эндпоинты для выдачи и возврата книг
@app.post("/borrow", response_model=BorrowedBookSchema, tags=["Выдача и возврат"])
async def borrow_book(data: BorrowSchema, session: SessionDep, user: UserDep):
    book = await session.get(BookModel, data.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")
    if book.copies <= 0:
        raise HTTPException(status_code=400, detail="Нет доступных экземпляров")

    active_borrows = await session.execute(
        select(func.count()).select_from(BorrowedBookModel)
        .filter_by(reader_id=data.reader_id, return_date=None)
    )
    if active_borrows.scalar() >= 3:
        raise HTTPException(status_code=400, detail="Читатель не может взять более 3 книг")

    reader = await session.get(ReaderModel, data.reader_id)
    if not reader:
        raise HTTPException(status_code=404, detail="Читатель не найден")

    borrow = BorrowedBookModel(book_id=data.book_id, reader_id=data.reader_id)
    book.copies -= 1
    session.add(borrow)
    await session.commit()
    await session.refresh(borrow)
    return borrow


@app.post("/return", response_model=BorrowedBookSchema, tags=["Выдача и возврат"])
async def return_book(data: BorrowSchema, session: SessionDep, user: UserDep):
    borrow = await session.execute(
        select(BorrowedBookModel)
        .filter_by(book_id=data.book_id, reader_id=data.reader_id, return_date=None)
    )
    borrow = borrow.scalars().first()
    if not borrow:
        raise HTTPException(status_code=400, detail="Книга не была выдана этому читателю или уже возвращена")

    book = await session.get(BookModel, data.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Книга не найдена")

    book.copies += 1
    borrow.return_date = datetime.utcnow()
    await session.commit()
    await session.refresh(borrow)
    return borrow


@app.get("/reader/{reader_id}/borrows", response_model=List[BorrowedBookSchema], tags=["Выдача и возврат"])
async def get_reader_borrows(reader_id: int, session: SessionDep, user: UserDep):
    reader = await session.get(ReaderModel, reader_id)
    if not reader:
        raise HTTPException(status_code=404, detail="Читатель не найден")

    borrows = await session.execute(
        select(BorrowedBookModel).filter_by(reader_id=reader_id, return_date=None)
    )
    return borrows.scalars().all()


# Инициализация базы данных
@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Запуск
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
