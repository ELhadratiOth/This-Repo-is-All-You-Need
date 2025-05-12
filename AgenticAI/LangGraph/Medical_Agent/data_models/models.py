import  re
from pydantic import BaseModel , Field , field_validator


# Field(pattern=""):
# ✅ Adds basic regex validation
# ❌ Does not allow custom error messages (it gives a generic error)
# ❌ Doesn’t support more complex or dynamic logic
# ✅ Is declarative and very clean

# @field_validator :
# You can:
# ✅ Write a custom error message
# ✅ Use additional logic (e.g., check if date is real using datetime.strptime)
# ✅ Support conditional rules like: allow future dates only, weekdays only, etc.


class DateTimeModel(BaseModel):
    date:str=Field(description="Properly formatted date", pattern=r'^\d{2}-\d{2}-\d{4} \d{2}:\d{2}$')
    
    @field_validator("date")
    def check_format_date(cls, v):
        print(cls.__name__) # This will print "DateTimeModel" it  is  like self
        if not re.match(r'^\d{2}-\d{2}-\d{4} \d{2}:\d{2}$', v):  # Ensures 'DD-MM-YYYY HH:MM' format
            raise ValueError("The date should be in format 'DD-MM-YYYY HH:MM'")
        return v



class DateModel(BaseModel):
    date: str = Field(description="Properly formatted date", pattern=r'^\d{2}-\d{2}-\d{4}$')
    @field_validator("date")
    def check_format_date(cls, v):
        if not re.match(r'^\d{2}-\d{2}-\d{4}$', v):  # Ensures DD-MM-YYYY format
            raise ValueError("The date must be in the format 'DD-MM-YYYY'")
        return v
     
class IdentificationNumberModel(BaseModel):
    id: int = Field(description="Identification number (7 or 8 digits long)")
    @field_validator("id")
    def check_format_id(cls, v):
        if not re.match(r'^\d{7,8}$', str(v)):  # Convert to string before matching
            raise ValueError("The ID number should be a 7 or 8-digit number")
        return v
