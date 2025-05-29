# %%
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import os

# %%
print("Путь к шаблонам:", os.path.abspath('templates'))
print("Содержимое папки templates:", os.listdir('templates'))

# %%
app = Flask(__name__)

try:
    model = load_model('best_composite_model.keras')
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    prediction = None
    
    if request.method == 'POST':
        if model is None:
            error = "Модель не загружена"
        else:
            try:
                # Получаем данные из формы
                input_data = np.array([[
                float(request.form.get('density', 0)),
                float(request.form.get('elastic_modulus', 0)),
                float(request.form.get('hardener_amount', 0)),
                float(request.form.get('epoxy_group', 0)),
                float(request.form.get('flash_point', 0)),
                float(request.form.get('surface_density', 0)),
                float(request.form.get('elastic_modulus_tension', 0)),
                float(request.form.get('tensile_strength', 0)),
                float(request.form.get('resin_consumption', 0)),
                float(request.form.get('angle', 0)),
                float(request.form.get('step', 0)),
                float(request.form.get('density_weave', 0))
                ]], dtype=np.float32)
                
                prediction = float(model.predict(input_data)[0][0])
                
            except Exception as e:
                error = f"Ошибка предсказания: {str(e)}"
    
    return render_template('index.html', 
                        prediction=prediction,
                        error=error)

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Сервер остановлен")
    except Exception as e:
        print(f"Ошибка: {e}")


