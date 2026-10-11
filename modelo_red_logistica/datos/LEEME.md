# Carpeta de datos

Copie aquí los 9 archivos CSV (descomprima `Datos.rar`). Se aceptan también dentro de una subcarpeta
(por ejemplo `datos/Datos/SAP.csv`):

| Archivo | Separador | Uso |
|---|---|---|
| SAP.csv | `,` | Fuente principal: viajes, cliente, referencia, centro productor, CEDI, tipo (XI = traslado), importe, volumen |
| TMS.csv | `;` | Complemento por viaje: origen/destino real, fechas, m3 por viaje |
| maestro_nodos.csv | `;` | Plantas/CEDI, ciudad, coordenadas, estado (cierre) |
| info_cedi.csv | `;` | Posiciones de estiba y capacidades operativas |
| capacidad_plantas.csv | `;` | Capacidad de producción y familias por planta |
| parametros_plantas.csv | `;` | Costo fijo mensual por nodo |
| maestro_referencias.csv | `;` | Familia, planta antes/nueva, situación post cambio, cajas por estiba, m3 por caja |
| maestro_clientes.csv | `;` | Ciudad, departamento, zona comercial y coordenadas del cliente |
| homologacion_centros.csv | `;` | Homologación de centros SAP |

Estos archivos no se suben al repositorio (ver `.gitignore`).
