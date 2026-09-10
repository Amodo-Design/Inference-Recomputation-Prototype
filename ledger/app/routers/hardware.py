from app.models import Hardware
from app.routers import make_crud_router
from app.schemas import HardwareCreate, HardwareRead

router = make_crud_router(
    prefix="/hardware",
    tag="hardware",
    orm_model=Hardware,
    pk_attr="hardware_id",
    create_schema=HardwareCreate,
    read_schema=HardwareRead,
)
