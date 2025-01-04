import openai
import requests
import time
import os
import nltk
import logging
from fastapi import FastAPI, HTTPException, Query
from supabase import create_client, Client
from typing import List, Dict
from dotenv import load_dotenv

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

load_dotenv()

# Configurar logging si lo deseas
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Conexión a Supabase
SUPABASE_URL = "https://bcmtpgavppyuvshohcvw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJjbXRwZ2F2cHB5dXZzaG9oY3Z3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyOTg2MTE1MiwiZXhwIjoyMDQ1NDM3MTUyfQ.sgSKeE2YOmqxQMASoNxqD6nlixUvNHx8X-T31Ily7uI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Ahora NO tomamos OpenAI ni Leonardo de variables de entorno.
# Serán asignadas dinámicamente.

# Endpoints de Leonardo
generation_url = "https://cloud.leonardo.ai/api/rest/v1/generations"

# Datos de tu proyecto
BUCKET_NAME = "Storyboard"
tabla_escenas = "escenas"
guion_id = "d68aeb6b-2d1a-43fc-8e75-a18c03e828f4"

app = FastAPI()

# ----------------------------------------------------------------
#               Funciones de obtención y liberación de keys
# ----------------------------------------------------------------

from typing import Tuple

def get_api_key(api_type: str) -> Tuple[str, str]:
    """
    Obtiene la primera clave libre (boolean_active=False) para el tipo de API especificado
    (API1=OpenAI, API2=Leonardo), la marca como activa y la retorna junto a su ID.
    """
    try:
        resp = supabase.table("api_keys") \
            .select("*") \
            .eq("api", api_type) \
            .eq("boolean_active", False) \
            .limit(1) \
            .execute()
        if resp.data:
            key_data = resp.data[0]
            supabase.table("api_keys") \
                .update({"boolean_active": True}) \
                .eq("id", key_data["id"]) \
                .execute()
            return key_data["api_key"], key_data["id"]
        else:
            raise HTTPException(status_code=503, detail=f"No hay claves disponibles para {api_type}.")
    except Exception as e:
        logging.error(f"Error obteniendo clave para {api_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al obtener la clave para {api_type}.")

def release_api_key(key_id: str) -> None:
    """
    Libera la clave (la marca como boolean_active=False) dado su ID.
    """
    try:
        supabase.table("api_keys") \
                .update({"boolean_active": False}) \
                .eq("id", key_id) \
                .execute()
        logging.info(f"Clave con ID {key_id} liberada.")
    except Exception as e:
        logging.error(f"Error al liberar la clave {key_id}: {e}")

# ----------------------------------------------------------------
#               Funciones de obtención de datos en Supabase
# ----------------------------------------------------------------

def obtener_escenas_filtradas(tabla, guion_id):
    columnas_relevantes = [
        "id_escena", "numero_escena", "id_guion", "contenido", "Time_of_Day",
        "Primary_Location", "Secondary_Location", "Set_Requirements", "One_Line_Description",
        "Summary", "Characters_Involved", "Character_Interactions",
        "Action_Description", "Movement_and_Staging", "Lighting_Needs",
        "Props_and_Set_Decorations", "Costumes", "Makeup_and_Hair", "Vehicles", "Animals"
    ]
    try:
        query = supabase.table(tabla).select(','.join(columnas_relevantes)).eq("id_guion", guion_id)
        respuesta = query.execute()
        if respuesta.data:
            return respuesta.data
        else:
            print(f"No se encontraron datos para el guion con id {guion_id} en la tabla {tabla}.")
            return []
    except Exception as e:
        print(f"Error al obtener datos de Supabase: {e}")
        return []

def obtener_escenas_con_imagenes(guion_id: str) -> List[Dict]:
    """
    Obtiene las escenas filtradas por un id_guion específico que tengan una imagen_url.
    """
    try:
        respuesta = supabase.table("escenas") \
                            .select("id_escena, numero_escena, imagen_url") \
                            .eq("id_guion", guion_id) \
                            .execute()
        if respuesta.data:
            return respuesta.data
        else:
            print(f"No se encontraron escenas con imágenes para el guion con id {guion_id}.")
            return []
    except Exception as e:
        print(f"Error al obtener datos de Supabase: {e}")
        return []

# ----------------------------------------------------------------
#               Funciones de procesamiento de escenas
# ----------------------------------------------------------------

def procesar_escena_filtrada(escena_dict):
    descripcion = f"Escena {escena_dict.get('numero_escena', '')}: {escena_dict.get('One_Line_Description', '')}\n\n"
    descripcion += f"Hora del día: {escena_dict.get('Time_of_Day', '')}\n"
    descripcion += f"Ubicación principal: {escena_dict.get('Primary_Location', '')}\n"
    descripcion += f"Ubicación secundaria: {escena_dict.get('Secondary_Location', '')}\n\n"
    descripcion += f"Resumen: {escena_dict.get('Summary', '')}\n\n"
    descripcion += f"Personajes involucrados: {escena_dict.get('Characters_Involved', '')}\n"
    descripcion += f"Interacciones entre personajes: {escena_dict.get('Character_Interactions', '')}\n\n"
    descripcion += f"Acciones importantes: {escena_dict.get('Action_Description', '')}\n"
    descripcion += f"Requisitos del set: {escena_dict.get('Set_Requirements', '')}\n"
    descripcion += f"Decoraciones y utilería: {escena_dict.get('Props_and_Set_Decorations', '')}\n\n"
    descripcion += f"Vestuario: {escena_dict.get('Costumes', '')}\n"
    descripcion += f"Maquillaje y peinado: {escena_dict.get('Makeup_and_Hair', '')}\n\n"
    descripcion += f"Iluminación: {escena_dict.get('Lighting_Needs', '')}\n"
    descripcion += f"Vehículos: {escena_dict.get('Vehicles', '')}\n"
    descripcion += f"Animales: {escena_dict.get('Animals', '')}\n"
    return descripcion.strip()

def obtener_url_presignada_imagen(id_escena: int) -> str:
    """
    Recupera la URL presignada almacenada en imagen_url para la escena especificada.
    """
    try:
        resultado = supabase.table("escenas").select("imagen_url").eq("id_escena", id_escena).execute()
        if resultado.data and len(resultado.data) > 0:
            signed_url = resultado.data[0].get("imagen_url")
            return signed_url
    except Exception as e:
        print(f"Error al recuperar la URL presignada para la escena {id_escena}: {e}")
    return None

# ----------------------------------------------------------------
#           Funciones para generación de prompts en OpenAI
# ----------------------------------------------------------------

def generar_prompt_con_reintento(descripcion, intentos=3):
    """
    Genera un prompt para Leonardo usando la API de OpenAI (API1).
    """
    for i in range(intentos):
        openai_key_id = None
        try:
            # --- Tomar clave dinámica de OpenAI (API1)
            openai_key, openai_key_id = get_api_key("API1")
            openai.api_key = openai_key

            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Ajusta el modelo si lo requieres
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto en generación de prompts para modelos de IA que generan imágenes reales en inglés, "
                            "especializado en Leonardo.ai. Tu objetivo es crear prompts claros, concisos y detallados "
                            "que ayuden a Leonardo.ai a producir imágenes realistas y coherentes. "
                            "Asegúrate de incluir detalles clave de la escena: época, hora del día, ubicación, "
                            "personajes con descripciones físicas y vestuario, acciones importantes, objetos relevantes "
                            "y cualquier otro elemento visual clave que defina la atmósfera. "
                            "Mantén un estilo realista, evita contenido inapropiado o violento, y cumple con las políticas de contenido de Leonardo.ai."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Descripción de la escena:\n\n{descripcion}\n\n"
                            "Por favor, genera un prompt para Leonardo.ai basado en esta descripción. "
                            "El prompt debe ser muy conciso (máximo 500 caracteres) y reflejar fielmente la escena. "
                            "Incluye personajes, acciones, ambientación (época, hora del día, ubicación), "
                            "objetos clave y detalles visuales importantes. "
                            "El estilo debe ser realista, como una fotografía real o una escena cinematográfica. "
                            "Evita redundancias, lenguaje abstracto o ambigüedades. "
                            "No incluyas contenido inapropiado ni de sangre excesiva para cumplir con las políticas de Leonardo.ai."
                        )
                    }
                ],
                temperature=0.5,
                max_tokens=500
            )
            return response['choices'][0]['message']['content'].strip()

        except Exception as e:
            print(f"Error al generar el prompt (intento {i+1}/{intentos}): {e}")
            time.sleep(2)

        finally:
            # --- Liberar la clave de OpenAI
            if openai_key_id:
                release_api_key(openai_key_id)

    return None

def validar_prompt(prompt, max_length=1500):
    if len(prompt) > max_length:
        print(f"El prompt excede el límite de {max_length} caracteres. Será recortado.")
        return prompt[:max_length]
    return prompt

def validar_contenido_prompt(prompt):
    """
    Valida el contenido con la API de Moderation de OpenAI (API1).
    """
    openai_key_id = None
    try:
        # --- Tomar clave de OpenAI para Moderation
        openai_key, openai_key_id = get_api_key("API1")
        openai.api_key = openai_key

        response = openai.Moderation.create(input=prompt)
        resultados = response["results"][0]
        if resultados["flagged"]:
            print("El prompt generado contiene contenido inapropiado según la API de moderación.")
            return False
        return True

    except Exception as e:
        print(f"Error al validar contenido del prompt: {e}")
        return False

    finally:
        # --- Liberar la clave
        if openai_key_id:
            release_api_key(openai_key_id)

# ----------------------------------------------------------------
#         Funciones para comunicación con Leonardo (API2)
# ----------------------------------------------------------------

def leonardo_post_generation(payload_inicial: dict):
    """
    Envía la solicitud de generación de imagen a Leonardo (API2), usando clave dinámica.
    Retorna (response.json(), key_id) para manejar su liberación en un nivel superior.
    """
    leo_key_id = None
    try:
        leo_key, leo_key_id = get_api_key("API2")
        # Armamos cabecera en cada llamada
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {leo_key}"
        }
        generation_response = requests.post(generation_url, json=payload_inicial, headers=headers)
        return generation_response, leo_key_id

    except Exception as e:
        logging.error(f"Error al enviar generación a Leonardo: {e}")
        return None, leo_key_id

def leonardo_check_status(generation_id: str):
    """
    Verifica el estado de la generación en Leonardo (API2), usando clave dinámica.
    Retorna (response.json(), key_id).
    """
    leo_key_id = None
    try:
        leo_key, leo_key_id = get_api_key("API2")
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {leo_key}"
        }
        status_url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}"
        status_response = requests.get(status_url, headers=headers)
        return status_response, leo_key_id
    except Exception as e:
        logging.error(f"Error al verificar estado en Leonardo: {e}")
        return None, leo_key_id

# ----------------------------------------------------------------
#              Función principal de procesamiento
# ----------------------------------------------------------------

def procesar_y_generar_imagenes():
    """
    Procesa las escenas, genera prompts (vía OpenAI), crea imágenes (vía Leonardo),
    sube las imágenes a Supabase y actualiza la base de datos con las URLs presignadas.
    """
    escenas = obtener_escenas_filtradas(tabla_escenas, guion_id)
    escenas.sort(key=lambda d: d.get("numero_escena", float("inf")))

    resultados = []
    if not escenas:
        print("No hay escenas disponibles para procesar.")
        actualizar_lookbook()
        return resultados

    for escena_dict in escenas:
        escena_id = escena_dict.get('id_escena')
        descripcion_escena = procesar_escena_filtrada(escena_dict)
        print(f"\nProcesando escena {escena_id}: {descripcion_escena}")

        # --- 1) Generar prompt con GPT (OpenAI)
        prompt_inicial = generar_prompt_con_reintento(descripcion_escena)
        if not prompt_inicial:
            print(f"No se pudo generar el prompt para escena {escena_id}.")
            continue

        # --- 2) Validar contenido del prompt
        if not validar_contenido_prompt(prompt_inicial):
            print(f"El prompt de la escena {escena_id} fue marcado como inapropiado.")
            continue

        # --- 3) Validar longitud (opcional)
        prompt_inicial = validar_prompt(prompt_inicial)
        print("Prompt inicial validado:")
        print(prompt_inicial)

        # --- 4) Llamada a Leonardo (API2) para iniciar generación
        payload_inicial = {
            "height": 768,
            "width": 768,
            "alchemy": True,
            "prompt": prompt_inicial,
            "num_images": 1,
            "negative_prompt": "",
            "modelId": "aa77f04e-3eec-4034-9c07-d0f619684628",
            "presetStyle": "STOCK_PHOTO",
            "photoReal": True,
            "photoRealVersion": "v2"
        }

        gen_resp, leo_key_id = leonardo_post_generation(payload_inicial)
        if not gen_resp or gen_resp.status_code != 200:
            print(f"Error al generar la imagen para escena {escena_id}.")
            # Liberar la clave de Leonardo
            if leo_key_id:
                release_api_key(leo_key_id)
            continue

        generation_data_inicial = gen_resp.json()
        generation_id_inicial = generation_data_inicial.get("sdGenerationJob", {}).get("generationId")
        if not generation_id_inicial:
            print(f"No se obtuvo generationId para la escena {escena_id}.")
            if leo_key_id:
                release_api_key(leo_key_id)
            continue

        # --- 5) Verificar estado de la generación (hasta 30 reintentos)
        max_retries = 30
        for retry_count in range(max_retries):
            time.sleep(5)
            stat_resp, stat_key_id = leonardo_check_status(generation_id_inicial)
            # Liberar inmediatamente la key que se usó
            if stat_key_id:
                release_api_key(stat_key_id)

            if not stat_resp or stat_resp.status_code != 200:
                print(f"Error al verificar estado (escena {escena_id}): {stat_resp and stat_resp.status_code}")
                break

            status_data_inicial = stat_resp.json()
            generation_status = status_data_inicial.get("generations_by_pk", {}).get("status")
            print(f"Estado de la generación: {generation_status}")

            if generation_status == "COMPLETE":
                print(f"La generación de la imagen para escena {escena_id} está completa.")
                break
            elif generation_status == "FAILED":
                print(f"La generación para escena {escena_id} ha fallado.")
                break
            else:
                print("La imagen aún se está generando. Esperando...")

            if retry_count == max_retries - 1:
                print(f"Tiempo de espera agotado para la escena {escena_id}.")
                break

        # Ahora que la generación está (teóricamente) completa o falló,
        # podemos liberar la key inicial de leonardo (si no la liberamos antes).
        if leo_key_id:
            release_api_key(leo_key_id)

        # --- 6) Si se completó, descargar la imagen y subirla
        if status_data_inicial:
            gen_images = status_data_inicial.get("generations_by_pk", {}).get("generated_images", [])
            if not gen_images:
                print(f"No se encontraron imágenes generadas para la escena {escena_id}.")
                continue

            image_data_inicial = gen_images[0]
            image_url_inicial = image_data_inicial["url"]
            print(f"Descargando imagen inicial desde {image_url_inicial}...")

            image_response_inicial = requests.get(image_url_inicial)
            if image_response_inicial.status_code == 200:
                image_data = image_response_inicial.content
                print("Imagen descargada en memoria.")

                # Subir la imagen al bucket privado
                archivo_remoto = f"escenas/guion_{guion_id}_escena_{escena_id}_imagen_inicial.jpg"
                try:
                    supabase.storage.from_(BUCKET_NAME).upload(archivo_remoto, image_data)
                    print(f"Imagen subida correctamente a {BUCKET_NAME}/{archivo_remoto}")

                    # Generar una URL presignada con expiración de 1 año (31536000 s)
                    signed_url_resp = supabase.storage.from_(BUCKET_NAME).create_signed_url(archivo_remoto, 31536000)
                    if signed_url_resp.get("error"):
                        print(f"Error al generar la URL presignada para la escena {escena_id}: {signed_url_resp['error']}")
                        continue
                    signed_url = signed_url_resp.get("signedURL")
                    if not signed_url:
                        print(f"No se pudo obtener la URL presignada para la escena {escena_id}.")
                        continue

                    # Actualizar la base de datos con la URL presignada
                    update_result = supabase.table(tabla_escenas).update({"imagen_url": signed_url}).eq("id_escena", escena_id).execute()
                    if update_result.error:
                        print(f"Error al actualizar imagen_url para la escena {escena_id}: {update_result.error}")
                    else:
                        print(f"URL presignada almacenada correctamente para la escena {escena_id}")
                except Exception as e:
                    print(f"Error al subir la imagen a Supabase Storage: {e}")
                    continue
            else:
                print(f"Error al descargar la imagen inicial: {image_response_inicial.status_code}")
                continue

        # Obtener la URL presignada (ya está almacenada en la columna imagen_url)
        url_imagen = obtener_url_presignada_imagen(escena_id)
        resultados.append({
            "id_escena": escena_id,
            "numero_escena": escena_dict.get("numero_escena"),
            "imagen_url_presignada": url_imagen
        })

    # Después de procesar todas las escenas
    actualizar_lookbook()
    return resultados

# ----------------------------------------------------------------
#       Función de actualización de 'Lookbook' en la tabla
# ----------------------------------------------------------------

def actualizar_lookbook():
    """
    Revisa las puntuaciones de arco dramático e intensidad emocional en la tabla 'escenas'
    y asigna 'si' o 'no' en la columna 'Lookbook' dependiendo de si ambas puntuaciones superan 0.8.
    """
    try:
        response = supabase.table("escenas") \
                           .select("id_escena, puntuacion_arco_dramatico, puntuacion_intensidad_emocional") \
                           .eq("id_guion", guion_id) \
                           .execute()
        escenas = response.data
    except Exception as e:
        print(f"Error al obtener las escenas: {e}")
        return
    
    if not escenas:
        print("No se encontraron escenas para el guion especificado.")
        return
    
    for escena in escenas:
        id_escena = escena.get("id_escena")
        puntuacion_arco = escena.get("puntuacion_arco_dramatico")
        puntuacion_intensidad = escena.get("puntuacion_intensidad_emocional")
        
        # Verificar que las puntuaciones existan y sean numéricas
        if puntuacion_arco is not None and puntuacion_intensidad is not None:
            try:
                puntuacion_arco = float(puntuacion_arco)
                puntuacion_intensidad = float(puntuacion_intensidad)
            except ValueError:
                print(f"Valores no numéricos para la escena {id_escena}. Asignando 'no' por defecto.")
                lookbook = "no"
            else:
                if puntuacion_arco > 0.8 and puntuacion_intensidad > 0.8:
                    lookbook = "si"
                else:
                    lookbook = "no"
        else:
            print(f"Faltan puntuaciones para la escena {id_escena}. Asignando 'no' por defecto.")
            lookbook = "no"
        
        # Actualizar la columna 'Lookbook' en la tabla 'escenas'
        try:
            update_response = supabase.table("escenas") \
                                       .update({"Lookbook": lookbook}) \
                                       .eq("id_escena", id_escena) \
                                       .execute()
            if update_response.status_code in range(200, 300):
                print(f"Escena {id_escena} actualizada con Lookbook: {lookbook}")
            else:
                print(f"Error al actualizar Lookbook para escena {id_escena}. "
                    f"status_code={update_response.status_code}, "
                    f"status_text={update_response.status_text}, "
                    f"data={update_response.data}")
        except Exception as e:
            print(f"Error al actualizar Lookbook para escena {id_escena}: {e}")

# ----------------------------------------------------------------
#                     Endpoints de FastAPI
# ----------------------------------------------------------------

@app.get("/procesar_escenas")
def procesar_escenas():
    """
    Endpoint que procesa las escenas, genera imágenes y devuelve las URLs presignadas.
    """
    resultados = procesar_y_generar_imagenes()
    return {"escenas": resultados}

@app.get("/obtener_imagenes")
def obtener_imagenes(guion_id: str = Query(..., description="ID del guion")):
    """
    Endpoint que obtiene las escenas con imágenes y devuelve sus URLs presignadas.
    Ejemplo: GET /obtener_imagenes?guion_id=<id_guion>
    """
    escenas = obtener_escenas_con_imagenes(guion_id)
    # Ordenar por numero_escena
    escenas.sort(key=lambda d: d.get("numero_escena", float("inf")))

    resultados = []
    for escena in escenas:
        escena_id = escena.get("id_escena")
        imagen_url = escena.get("imagen_url")

        if imagen_url:
            resultados.append({
                "id_escena": escena_id,
                "numero_escena": escena.get("numero_escena"),
                "url_presignada": imagen_url
            })
        else:
            print(f"La escena {escena_id} no tiene imagen_url guardada.")

    return {"escenas": resultados}

@app.get("/obtener_imagen/{id_escena}")
def obtener_imagen(id_escena: int):
    """
    Endpoint para obtener la URL presignada de la imagen de una escena específica.
    Ejemplo: GET /obtener_imagen/10
    """
    respuesta = supabase.table("escenas").select("imagen_url").eq("id_escena", id_escena).execute()
    if respuesta.data and len(respuesta.data) > 0:
        imagen_url = respuesta.data[0].get("imagen_url")
        if imagen_url:
            return {"id_escena": id_escena, "url_presignada": imagen_url}
        else:
            raise HTTPException(status_code=404, detail="La escena no tiene imagen_url guardada.")
    else:
        raise HTTPException(status_code=404, detail="No se encontró la escena.")

@app.get("/escenas_lookbook_si")
def obtener_escenas_lookbook_si(id_guion: str = Query(..., description="ID del guion")):
    """
    Endpoint que devuelve las escenas del guion indicado cuyo campo 'Lookbook' sea 'si'.
    """
    try:
        columnas_relevantes = "id_escena, numero_escena, imagen_url"
        respuesta = supabase.table("escenas").select(columnas_relevantes).eq("Lookbook", "si").eq("id_guion", id_guion).execute()
        escenas = respuesta.data
        if escenas:
            return {"escenas": escenas}
        else:
            return {"escenas": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener las escenas: {e}")


