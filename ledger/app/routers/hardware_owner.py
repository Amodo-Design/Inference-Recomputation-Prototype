from app.models import HardwareOwner
from app.routers import make_crud_router
from app.schemas import HardwareOwnerCreate, HardwareOwnerRead

router = make_crud_router(
    prefix="/hardware-owners",
    tag="hardware_owner",
    orm_model=HardwareOwner,
    pk_attr="owner_id",
    create_schema=HardwareOwnerCreate,
    read_schema=HardwareOwnerRead,
)
