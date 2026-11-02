import { FileJson, Upload } from 'lucide-react';
import { useRef, useState, type DragEvent } from 'react';

interface DropZoneProps {
  label: string;
  hint: string;
  accept: string;
  required?: boolean;
  fileName?: string | undefined;
  disabled?: boolean;
  onFile: (file: File) => void;
}

export function DropZone({ label, hint, accept, required, fileName, disabled, onFile }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const takeDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) onFile(file);
  };

  return (
    <div
      className={`group relative flex min-h-32 cursor-pointer flex-col justify-between rounded-2xl border p-4 transition-colors ${dragging ? 'border-lime bg-lime/10' : 'border-line bg-white/[0.025] hover:border-slate-500 hover:bg-white/[0.04]'} ${disabled ? 'pointer-events-none opacity-50' : ''}`}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label={`Choose ${label}`}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={takeDrop}
    >
      <input
        ref={inputRef}
        className="sr-only"
        type="file"
        accept={accept}
        disabled={disabled}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
          event.currentTarget.value = '';
        }}
      />
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="text-sm font-semibold text-slate-100">{label}</span>
          {required ? <span className="ml-2 rounded-full bg-lime/15 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-lime">Required</span> : null}
        </div>
        {fileName ? <FileJson aria-hidden="true" className="h-5 w-5 text-lime" /> : <Upload aria-hidden="true" className="h-5 w-5 text-muted transition-colors group-hover:text-slate-200" />}
      </div>
      <div>
        <p className="truncate text-sm text-slate-200">{fileName ?? 'Drop file here or browse'}</p>
        <p className="mt-1 text-xs leading-5 text-muted">{hint}</p>
      </div>
    </div>
  );
}
