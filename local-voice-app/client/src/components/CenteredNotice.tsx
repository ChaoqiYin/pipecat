export const CenteredNotice = ({
  title,
  hint,
  destructive = false,
}: {
  title: string;
  hint?: string;
  destructive?: boolean;
}) => (
  <div className="flex h-full items-center justify-center">
    <div className="text-center p-4 max-w-md">
      <div
        className={
          destructive
            ? 'text-destructive font-medium mb-2'
            : 'text-muted-foreground mb-2'
        }
      >
        {title}
      </div>
      {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
    </div>
  </div>
);
