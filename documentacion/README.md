# Documentación de SPELLFIND

`main.tex` es el documento principal; incorpora `plan_mejoras.tex`,
`bloque1.tex`, `bloque2.tex`, `bloque3.tex`, `bloque4.tex` y `evolucion.tex` con `\input`.
Compilar el proyecto completo desde esta carpeta para resolver esa inclusión.
Los siete archivos forman parte del mismo documento y deben conservarse juntos.

El documento distingue:

- Requisitos y capacidades actuales.
- Versiones históricas A/B/C y afirmaciones de los PDF originales.
- Cambios históricos 1–8 e implementación local P01–P16, con justificación,
  contrapartidas y verificaciones reales/humanas pendientes en P04/P10/P12/P16.
- Plan vigente P01–P20, en cinco bloques de cuatro actividades, con dependencias
  y criterios de aceptación.
- Referencias históricas 9–14 y E01–E16, enlazadas al plan vigente sin duplicar
  tareas. E14 se asignó a II/P07, E15 a II/P05/P08 y E16 a I/P01/P02; están
  corregidos con pruebas en `../validation/block3/`. La revisión original permanece
  en `../validation/revision_block2/`. P05–P09/P11 tienen pruebas locales; P01–P03 están verificados localmente;
  P04 tiene protocolo y simulación, con evaluación musical real pendiente.
- El siguiente bloque de implementación es P17–P20. P13–P15 están implementados;
  P16 tiene instrumentación y perfil local con modelos de control. P10/P12/P16
  siguen abiertos para calibración, comparación y carga real. No hay nueva
  compilación PDF certificada.

## Registro obligatorio para los siguientes bloques

En cada implementación, actualizar el README del proyecto, el estado de la
actividad en `plan_mejoras.tex` y el registro `evolucion.tex`:

1. Fecha y punto del plan al que responde.
2. Problema anterior y versión afectada.
3. Comportamiento nuevo y archivos modificados.
4. Por qué mejora una propiedad concreta y qué evidencia lo respalda.
5. Pruebas realmente ejecutadas, entorno y límites de la verificación.
6. Costes, pérdida de capacidades, compatibilidad y alternativas.
7. Nuevas tareas si el beneficio no se demuestra o queda una solución mejor.

No declarar calidad musical a partir de pruebas con modelos simulados, ni
velocidad o memoria a partir del número de llamadas exclusivamente. Mantener
separados los resultados históricos y las medidas nuevas. Documentar también
cuando una alternativa resulte peor o deba revertirse.

Para cerrar una actividad P, comprobar también qué referencias 9–14/E cubre.
Conservar todos los identificadores y registrar implementación,
fecha, evidencia y condición de aceptación satisfecha. No eliminar el diagnóstico
histórico ni presentarlo como estado actual. Los originales de los experimentos
permanecen en `../experiments/`.
# Bloque V

`main.tex` incorpora `bloque5.tex`: P17 snapshots y publicación, P18 decisión
sobre checkpoints/polling, P19 ingesta opcional y P20 validación integrada.
E17 retención, E18 entorno e integración reales y E19 modalidades opcionales
permanecen abiertos. La evidencia nueva se guarda en `../validation/block5/`.
Las comprobaciones estáticas de LaTeX no equivalen a compilación PDF.

# Correcciones de pendientes

`correcciones_pendientes.tex` registra la verificación conservadora de restricciones,
el retiro de decisiones antiguas sin evidencia y las herramientas de anotación
humana/perfil real. Está incluido en `main.tex`; la evidencia posterior se guarda
en `../validation/pending_fixes/`. Los estados históricos no equivalen al actual.

