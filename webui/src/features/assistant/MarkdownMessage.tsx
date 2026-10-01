import { useState, type ReactNode } from 'react';
import { Check, Copy } from 'lucide-react';

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="assistant-code">
      <div className="assistant-code-bar">
        <span>{lang || 'code'}</span>
        <button type="button" onClick={() => {
          void navigator.clipboard.writeText(code).then(() => {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1200);
          });
        }}>
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? '已复制' : '复制'}
        </button>
      </div>
      <pre><code>{code}</code></pre>
    </div>
  );
}

function inline(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = pattern.exec(text))) {
    if (match.index > last) nodes.push(text.slice(last, match.index));
    const token = match[0];
    if (token.startsWith('`')) {
      nodes.push(<code key={key} className="assistant-inline-code">{token.slice(1, -1)}</code>);
    } else if (token.startsWith('**')) {
      nodes.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else {
      const link = /\[([^\]]+)\]\(([^)]+)\)/.exec(token);
      nodes.push(<a key={key} href={link?.[2]} target="_blank" rel="noreferrer">{link?.[1]}</a>);
    }
    key += 1;
    last = match.index + token.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function renderBlocks(content: string): ReactNode[] {
  const lines = content.replace(/\r\n/g, '\n').split('\n');
  const out: ReactNode[] = [];
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^```(\w+)?\s*$/);
    if (fence) {
      const lang = fence[1] || '';
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith('```')) {
        body.push(lines[i]);
        i += 1;
      }
      out.push(<CodeBlock key={key} code={body.join('\n')} lang={lang} />);
      key += 1;
      i += 1;
      continue;
    }
    if (/^\s*[-*] /.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*] /.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*] /, ''));
        i += 1;
      }
      out.push(<ul key={key}>{items.map((item, idx) => <li key={idx}>{inline(item)}</li>)}</ul>);
      key += 1;
      continue;
    }
    if (line.startsWith('### ')) {
      out.push(<h4 key={key}>{inline(line.slice(4))}</h4>);
      key += 1;
      i += 1;
      continue;
    }
    if (line.startsWith('## ') || line.startsWith('# ')) {
      out.push(<h3 key={key}>{inline(line.replace(/^#+ /, ''))}</h3>);
      key += 1;
      i += 1;
      continue;
    }
    if (!line.trim()) {
      i += 1;
      continue;
    }
    const para: string[] = [line];
    i += 1;
    while (i < lines.length && lines[i].trim() && !lines[i].startsWith('```') && !/^\s*[-*] /.test(lines[i]) && !lines[i].startsWith('#')) {
      para.push(lines[i]);
      i += 1;
    }
    out.push(<p key={key}>{inline(para.join(' '))}</p>);
    key += 1;
  }
  return out;
}

export function MarkdownMessage({ content }: { content: string }) {
  return <div className="assistant-md">{renderBlocks(content || '')}</div>;
}
