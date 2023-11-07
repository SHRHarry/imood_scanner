set CUR_DIR=%CD%

call conda activate C:\Users\Owner\Desktop\Harry\POSTER_V2\poster_env
cd ..
call python manage.py runserver
cd %CUR_DIR%