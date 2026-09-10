from app.models import VerificationEvent
from app.routers import make_crud_router
from app.schemas import VerificationEventCreate, VerificationEventRead

router = make_crud_router(
    prefix="/verification-events",
    tag="verification_event",
    orm_model=VerificationEvent,
    pk_attr="id",
    create_schema=VerificationEventCreate,
    read_schema=VerificationEventRead,
)
