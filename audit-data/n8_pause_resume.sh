#!/bin/bash
# Пауза/возобновление кампаний n8 (CMA-лестница) и n10 (дожим k=44).
# Использование: ./n8_pause_resume.sh pause | resume | status
PATS='n8_cma44_ladder.py|n10_push44.py|multiprocessing.spawn|multiprocessing.resource_tracker'
PIDS=$(pgrep -f "$PATS" | tr '\n' ' ')
if [ -z "$PIDS" ]; then
    echo "кампании не запущены. Докатка n8: .venv/bin/python -m chromatic_research.campaigns.n8_cma44_ladder (пропустит готовые k);"
    echo "перезапуск n10: .venv/bin/python -m chromatic_research.campaigns.n10_push44"
    exit 0
fi
case "$1" in
    pause)  kill -STOP $PIDS 2>/dev/null; echo "SIGSTOP -> $PIDS" ;;
    resume) kill -CONT $PIDS 2>/dev/null; echo "SIGCONT -> $PIDS" ;;
    status) ps -o pid,stat,%cpu,etime,command -p $PIDS | sed 's|/opt/homebrew.*MacOS/Python|python|' ;;
    *) echo "использование: $0 pause|resume|status" ;;
esac
